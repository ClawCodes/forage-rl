"""Build external perturbation manifests from saved run-dataset artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from forage_rl.config import MAZE_SPECS_DIR
from forage_rl.environments import load_builtin_maze_spec


_PERTURBATION_MARKER = "_perturbed_"


@dataclass(frozen=True)
class ManifestArtifact:
    """Resolved output paths for one converted run dataset."""

    source_run_dataset: Path
    combined_trajectory_path: Path
    manifest_record: dict[str, Any]


def _metadata_path_for_run_dataset(run_dataset_path: Path) -> Path:
    return run_dataset_path.with_suffix(".json")


def _load_run_dataset_artifact(
    run_dataset_path: Path,
) -> tuple[str, list[np.ndarray], dict[str, Any]]:
    metadata_path = _metadata_path_for_run_dataset(run_dataset_path)
    if not metadata_path.exists():
        raise ValueError(
            f"Run dataset metadata is missing for {run_dataset_path.name}; expected {metadata_path.name}."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(run_dataset_path, allow_pickle=True) as payload:
        if "__transition_type__" not in payload:
            raise ValueError(
                f"Run dataset {run_dataset_path.name} is missing __transition_type__ metadata."
            )
        transition_type = str(payload["__transition_type__"].item())
        episode_keys = sorted(key for key in payload.files if key.startswith("episode_"))
        if not episode_keys:
            raise ValueError(
                f"Run dataset {run_dataset_path.name} contains no episode arrays."
            )
        episodes = [np.asarray(payload[key]) for key in episode_keys]
    return transition_type, episodes, metadata


def _infer_benchmark_maze_name(artifact_maze_name: str) -> str:
    if _PERTURBATION_MARKER in artifact_maze_name:
        return artifact_maze_name.split(_PERTURBATION_MARKER, maxsplit=1)[0]
    return artifact_maze_name


def _infer_perturbation_kind(
    artifact_maze_name: str,
    *,
    metadata: dict[str, Any],
    override: str | None,
) -> str:
    if override is not None:
        return override
    if _PERTURBATION_MARKER in artifact_maze_name:
        return artifact_maze_name.split(_PERTURBATION_MARKER, maxsplit=1)[1]
    if metadata.get("perturbation_id") is not None:
        return str(metadata["perturbation_id"])
    return "perturbation"


def _builtin_spec_path(name: str) -> Path | None:
    spec_path = MAZE_SPECS_DIR / f"{name}.toml"
    return spec_path.resolve() if spec_path.exists() else None


def _infer_perturbation_timestep(
    *,
    metadata: dict[str, Any],
    artifact_maze_name: str,
    episode_lengths: list[int],
    override: int | None,
) -> int:
    if override is not None:
        return int(override)

    if metadata.get("perturbation_episode") is not None:
        perturbation_episode = int(metadata["perturbation_episode"])
        if perturbation_episode < 0 or perturbation_episode > len(episode_lengths):
            raise ValueError(
                "perturbation_episode metadata is out of range for the run dataset, "
                f"got {perturbation_episode} with {len(episode_lengths)} episodes."
            )
        return int(sum(episode_lengths[:perturbation_episode]))

    spec_path = _builtin_spec_path(artifact_maze_name)
    if spec_path is None:
        raise ValueError(
            "Could not infer perturbation_timestep automatically because no built-in "
            f"maze spec exists for {artifact_maze_name!r}. Pass --perturbation-timestep."
        )

    spec = load_builtin_maze_spec(artifact_maze_name)
    if spec.perturbation is None:
        raise ValueError(
            "Could not infer perturbation_timestep automatically because "
            f"{artifact_maze_name!r} has no perturbation block. Pass --perturbation-timestep."
        )
    if len(episode_lengths) != 1:
        raise ValueError(
            "Could not infer perturbation_timestep automatically for a multi-episode run "
            "because the built-in perturbation time is local to one episode. "
            "Pass --perturbation-timestep explicitly."
        )
    return int(spec.perturbation.perturbation_time)


def _infer_perturbation_id(
    *,
    metadata: dict[str, Any],
    perturbation_timestep: int,
    perturbation_kind: str,
    override: str | None,
) -> str:
    if override is not None:
        return override
    if metadata.get("perturbation_id") is not None:
        return str(metadata["perturbation_id"])
    return f"{perturbation_kind}_t{perturbation_timestep}"


def _relative_to_manifest(path: Path, *, manifest_path: Path) -> str:
    return os.path.relpath(
        path.resolve(),
        start=manifest_path.parent.resolve(),
    )


def _manifest_record(
    *,
    source_run_dataset: Path,
    combined_trajectory_path: Path,
    metadata: dict[str, Any],
    artifact_maze_name: str,
    benchmark_maze_name: str,
    transition_type: str,
    episode_lengths: list[int],
    perturbation_kind: str,
    perturbation_id: str,
    perturbation_timestep: int,
    manifest_path: Path,
    benchmark_kind: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": source_run_dataset.stem,
        "agent": str(metadata["agent"]),
        "maze_name": benchmark_maze_name,
        "observable": metadata["observable"],
        "perturbation_kind": perturbation_kind,
        "perturbation_id": perturbation_id,
        "trajectory_path": _relative_to_manifest(
            combined_trajectory_path,
            manifest_path=manifest_path,
        ),
        "perturbation_timestep": perturbation_timestep,
        "episode_lengths": episode_lengths,
        "transition_type": transition_type,
        "horizon": int(metadata["horizon"]),
    }
    if metadata.get("context_mode") is not None:
        record["context_mode"] = str(metadata["context_mode"])
    if metadata.get("notes") is not None:
        record["notes"] = str(metadata["notes"])
    if benchmark_kind is not None:
        record["benchmark_kind"] = benchmark_kind

    spec_path = _builtin_spec_path(artifact_maze_name)
    if spec_path is not None and artifact_maze_name != benchmark_maze_name:
        record["benchmark_params"] = {
            "maze_spec_path": _relative_to_manifest(spec_path, manifest_path=manifest_path)
        }
    return record


def build_external_perturbation_manifest(
    *,
    input_run_datasets: list[str | Path],
    output_manifest: str | Path,
    perturbation_kind: str | None = None,
    perturbation_id: str | None = None,
    perturbation_timestep: int | None = None,
    benchmark_maze_name: str | None = None,
    benchmark_kind: str | None = None,
) -> list[ManifestArtifact]:
    """Convert saved run datasets into combined trajectories plus one manifest."""
    manifest_path = Path(output_manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts: list[ManifestArtifact] = []
    records: list[dict[str, Any]] = []
    for raw_path in input_run_datasets:
        run_dataset_path = Path(raw_path).expanduser().resolve()
        transition_type, episodes, metadata = _load_run_dataset_artifact(run_dataset_path)

        artifact_maze_name = str(metadata["maze_name"])
        resolved_benchmark_maze_name = (
            benchmark_maze_name
            if benchmark_maze_name is not None
            else _infer_benchmark_maze_name(artifact_maze_name)
        )
        episode_lengths = [int(len(episode)) for episode in episodes]
        resolved_perturbation_timestep = _infer_perturbation_timestep(
            metadata=metadata,
            artifact_maze_name=artifact_maze_name,
            episode_lengths=episode_lengths,
            override=perturbation_timestep,
        )
        resolved_perturbation_kind = _infer_perturbation_kind(
            artifact_maze_name,
            metadata=metadata,
            override=perturbation_kind,
        )
        resolved_perturbation_id = _infer_perturbation_id(
            metadata=metadata,
            perturbation_timestep=resolved_perturbation_timestep,
            perturbation_kind=resolved_perturbation_kind,
            override=perturbation_id,
        )

        combined_trajectory_path = manifest_path.parent / f"{run_dataset_path.stem}_combined.npy"
        combined = np.concatenate(episodes, axis=0)
        np.save(combined_trajectory_path, combined)

        record = _manifest_record(
            source_run_dataset=run_dataset_path,
            combined_trajectory_path=combined_trajectory_path,
            metadata=metadata,
            artifact_maze_name=artifact_maze_name,
            benchmark_maze_name=resolved_benchmark_maze_name,
            transition_type=transition_type,
            episode_lengths=episode_lengths,
            perturbation_kind=resolved_perturbation_kind,
            perturbation_id=resolved_perturbation_id,
            perturbation_timestep=resolved_perturbation_timestep,
            manifest_path=manifest_path,
            benchmark_kind=benchmark_kind,
        )
        records.append(record)
        artifacts.append(
            ManifestArtifact(
                source_run_dataset=run_dataset_path,
                combined_trajectory_path=combined_trajectory_path,
                manifest_record=record,
            )
        )

    manifest_path.write_text(json.dumps({"runs": records}, indent=2), encoding="utf-8")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an external perturbation analysis manifest from run datasets."
    )
    parser.add_argument(
        "--input-run-datasets",
        nargs="+",
        required=True,
        help="One or more saved run_dataset .npz files to convert.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Where to write the manifest JSON. Combined trajectories are written alongside it.",
    )
    parser.add_argument(
        "--perturbation-kind",
        default=None,
        help="Optional perturbation kind override for all runs.",
    )
    parser.add_argument(
        "--perturbation-id",
        default=None,
        help="Optional perturbation id override for all runs.",
    )
    parser.add_argument(
        "--perturbation-timestep",
        type=int,
        default=None,
        help="Optional global perturbation timestep override for all runs.",
    )
    parser.add_argument(
        "--benchmark-maze-name",
        default=None,
        help="Optional benchmark maze family override (e.g. full_one_way).",
    )
    parser.add_argument(
        "--benchmark-kind",
        choices=["true_mvt", "fo_oracle"],
        default=None,
        help="Optional benchmark kind override stored in the manifest.",
    )

    args = parser.parse_args()
    artifacts = build_external_perturbation_manifest(
        input_run_datasets=args.input_run_datasets,
        output_manifest=args.output_manifest,
        perturbation_kind=args.perturbation_kind,
        perturbation_id=args.perturbation_id,
        perturbation_timestep=args.perturbation_timestep,
        benchmark_maze_name=args.benchmark_maze_name,
        benchmark_kind=args.benchmark_kind,
    )
    print(f"Wrote {len(artifacts)} run(s) to {Path(args.output_manifest).expanduser().resolve()}")


if __name__ == "__main__":
    main()
