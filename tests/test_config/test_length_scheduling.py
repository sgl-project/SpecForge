"""Length policies are opt-in and restricted to validated objective paths."""

import copy

import pytest
from pydantic import ValidationError

from specforge.config import Config
from tests.test_config.test_schema import MINIMAL, _online_payload


def test_default_preserves_both_legacy_paths():
    for payload in (copy.deepcopy(MINIMAL), _online_payload()):
        training = Config.model_validate(payload).training
        assert not training.length_aware_scheduling
        assert training.length_bucket_size == 0


def test_offline_dflash_family_bucketing():
    payload = copy.deepcopy(MINIMAL)
    payload["training"] = {
        "strategy": "dflash",
        "length_bucket_size": 32,
        "batch_size": 2,
    }
    config = Config.model_validate(payload)
    assert config.training.length_bucket_size == 32
    payload["training"]["length_bucket_size"] = -1
    with pytest.raises(ValidationError):
        Config.model_validate(payload)


@pytest.mark.parametrize(
    "strategy,batch_size", [("eagle3", 1), ("dflash", 1), ("dflash", 4)]
)
def test_online_supported_objectives(strategy, batch_size):
    payload = _online_payload(strategy)
    payload["training"].update(length_aware_scheduling=True, batch_size=batch_size)
    assert Config.model_validate(payload).training.length_aware_scheduling


def test_unsupported_objective_regrouping_fails_before_launch():
    offline = copy.deepcopy(MINIMAL)
    offline["training"] = {"length_bucket_size": 8, "batch_size": 2}
    with pytest.raises(ValidationError, match="padded-length loss"):
        Config.model_validate(offline)
    online = _online_payload()
    online["training"].update(length_aware_scheduling=True, batch_size=2)
    with pytest.raises(ValidationError, match="batch_size=1"):
        Config.model_validate(online)


def test_policy_cannot_silently_do_nothing_in_wrong_mode():
    offline = copy.deepcopy(MINIMAL)
    offline["training"] = {"length_aware_scheduling": True}
    with pytest.raises(ValidationError, match="requires online"):
        Config.model_validate(offline)
    online = _online_payload("dflash")
    online["training"]["length_bucket_size"] = 8
    with pytest.raises(ValidationError, match="requires offline"):
        Config.model_validate(online)
