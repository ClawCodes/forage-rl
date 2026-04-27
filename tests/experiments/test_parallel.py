from forage_rl.agents.registry import Agent, EvaluatorSpec
from forage_rl.experiments import parallel


def test_build_torch_batches_keeps_cpu_batches_combined(monkeypatch) -> None:
    monkeypatch.setattr(parallel, "resolve_device", lambda device: "cpu")

    items = [Agent.MBRL, Agent.DQN, Agent.QLearning]

    assert parallel.build_torch_batches(items, device="auto") == [
        (items, True, "mixed CPU")
    ]


def test_build_torch_batches_splits_non_cpu_workloads(monkeypatch) -> None:
    monkeypatch.setattr(parallel, "resolve_device", lambda device: "cuda")

    items = [
        EvaluatorSpec(agent=Agent.MBRL),
        EvaluatorSpec(agent=Agent.DQN),
        EvaluatorSpec(agent=Agent.QLearning),
    ]

    assert parallel.build_torch_batches(items, device="auto") == [
        ([items[0], items[2]], False, "CPU-only"),
        ([items[1]], True, "neural"),
    ]


def test_build_torch_batches_keeps_cpu_only_batches_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(parallel, "resolve_device", lambda device: "cpu")

    items = [Agent.MBRL, Agent.QLearning]

    assert parallel.build_torch_batches(items, device="auto") == [
        (items, False, "CPU-only")
    ]
