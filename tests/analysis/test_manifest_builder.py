import json

import numpy as np

from forage_rl.analysis.manifest_builder import build_external_perturbation_manifest


def _write_run_dataset(
    path,
    *,
    episode_lengths: list[int],
    transition_type: str = "TimedTransition",
    metadata: dict,
) -> None:
    payload: dict[str, object] = {
        "__transition_type__": np.array(transition_type, dtype=np.str_),
    }
    for episode_index, length in enumerate(episode_lengths):
        payload[f"episode_{episode_index:05d}"] = np.zeros((length, 5), dtype=float)
    np.savez(path, **payload)
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")


def test_build_manifest_infers_single_episode_midpoint_from_builtin_spec(tmp_path):
    run_dataset_path = tmp_path / "full_one_way_perturbed_detour_FO_q_learning_run_dataset_0.npz"
    _write_run_dataset(
        run_dataset_path,
        episode_lengths=[1000],
        metadata={
            "container_type": "run_dataset",
            "agent": "q_learning",
            "maze_name": "full_one_way_perturbed_detour",
            "observable": True,
            "horizon": 1000,
            "num_episodes": 1,
            "num_transitions": 1000,
        },
    )

    manifest_path = tmp_path / "manifest.json"
    artifacts = build_external_perturbation_manifest(
        input_run_datasets=[run_dataset_path],
        output_manifest=manifest_path,
    )

    assert len(artifacts) == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert run["maze_name"] == "full_one_way"
    assert run["perturbation_kind"] == "detour"
    assert run["perturbation_timestep"] == 500
    assert run["episode_lengths"] == [1000]
    assert run["benchmark_params"]["maze_spec_path"].endswith(
        "forage_rl/environments/maze_specs/full_one_way_perturbed_detour.toml"
    )
    combined = np.load(tmp_path / run["trajectory_path"])
    assert combined.shape == (1000, 5)


def test_build_manifest_uses_perturbation_episode_metadata_when_available(tmp_path):
    run_dataset_path = tmp_path / "simple_FO_sr_mb_perturb_decay_swap_run_dataset_0.npz"
    _write_run_dataset(
        run_dataset_path,
        episode_lengths=[100, 100],
        metadata={
            "container_type": "perturbation_run_dataset",
            "agent": "sr_mb",
            "maze_name": "simple",
            "observable": True,
            "horizon": 100,
            "num_episodes": 2,
            "num_transitions": 200,
            "perturbation_episode": 1,
            "perturbation_id": "decay_swap",
        },
    )

    manifest_path = tmp_path / "simple_manifest.json"
    build_external_perturbation_manifest(
        input_run_datasets=[run_dataset_path],
        output_manifest=manifest_path,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert run["maze_name"] == "simple"
    assert run["perturbation_timestep"] == 100
    assert run["perturbation_kind"] == "decay_swap"
    assert run["perturbation_id"] == "decay_swap"
    combined = np.load(tmp_path / run["trajectory_path"])
    assert combined.shape == (200, 5)
