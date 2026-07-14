"""Default provider hooks shared by built-in registrations."""

from __future__ import annotations


def empty_options(_config):
    return {}


def empty_resume_contract(_config, _draft_model, _training_model):
    return {}


def one_loss_token(_config, _draft_config=None):
    return 1


def online_needs_input_tools(config, _draft_model):
    return config.mode == "online"


__all__ = [
    "empty_options",
    "empty_resume_contract",
    "one_loss_token",
    "online_needs_input_tools",
]
