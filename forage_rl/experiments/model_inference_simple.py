"""Simple model inference test on saved run datasets."""

import numpy as np

from forage_rl.agents.registry import Agent, EvaluatorSpec
from forage_rl.experiments.model_inference import evaluate_run_dataset
from forage_rl.utils import get_run_count, load_run_dataset


def _total_for(run_results: dict[EvaluatorSpec, np.ndarray], agent: Agent) -> float:
    for evaluator, values in run_results.items():
        if evaluator.agent == agent:
            return float(np.sum(values))
    raise ValueError(f"Missing evaluator results for {agent.value}.")


def run_simple_inference(
    maze_name: str = "simple",
    mbrl_file_id: int = 0,
    qlearning_file_id: int = 0,
    observable: bool = True,
):
    """Run simple inference on one saved run dataset per source agent."""
    mbrl_count = get_run_count(Agent.MBRL, maze_name, observable)
    ql_count = get_run_count(Agent.QLearning, maze_name, observable)

    if mbrl_count == 0 or ql_count == 0:
        print("No run datasets found. Run generate_trajectories.py first.")
        return

    evaluators = [Agent.MBRL, Agent.QLearning]

    print("=" * 60)
    print("Evaluating transitions from MBRL simulation")
    print("=" * 60)

    run_dataset = load_run_dataset(Agent.MBRL, mbrl_file_id, maze_name, observable)
    run_results = evaluate_run_dataset(
        run_dataset,
        maze_name=maze_name,
        evaluators=evaluators,
        observable=observable,
    )
    mb_total = _total_for(run_results, Agent.MBRL)
    ql_total = _total_for(run_results, Agent.QLearning)
    print(f"MBRL log-likelihood: {mb_total:.4f}")
    print(f"Q-learning log-likelihood: {ql_total:.4f}")

    if mb_total > ql_total:
        print("Result: MBRL explains the data better")
    else:
        print("Result: Q-learning explains the data better")

    print("\n" + "=" * 60)
    print("Evaluating transitions from Q-learning simulation")
    print("=" * 60)

    run_dataset = load_run_dataset(
        Agent.QLearning,
        qlearning_file_id,
        maze_name,
        observable,
    )
    run_results = evaluate_run_dataset(
        run_dataset,
        maze_name=maze_name,
        evaluators=evaluators,
        observable=observable,
    )
    mb_total = _total_for(run_results, Agent.MBRL)
    ql_total = _total_for(run_results, Agent.QLearning)
    print(f"MBRL log-likelihood: {mb_total:.4f}")
    print(f"Q-learning log-likelihood: {ql_total:.4f}")

    if ql_total > mb_total:
        print("Result: Q-learning explains the data better")
    else:
        print("Result: MBRL explains the data better")


if __name__ == "__main__":
    run_simple_inference()
