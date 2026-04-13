"""Backward-compatible DRQN alias for the canonical LSTM recurrent agent."""

from forage_rl.agents.recurrent import LSTMAgent as DRQNAgent

__all__ = ["DRQNAgent"]
