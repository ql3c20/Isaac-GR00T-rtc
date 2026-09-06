from types import SimpleNamespace

import pytest
from scripts.deployment.trt_model_forward import _prefix_rtc_timestep_bucket


def _head(mode: str, buckets: int = 1000):
    return SimpleNamespace(
        _prefix_rtc_timestep_mode=mode,
        num_timestep_buckets=buckets,
        config=SimpleNamespace(prefix_rtc_timestep_mode=mode),
    )


def test_prefix_rtc_legacy_zero_uses_bucket_zero():
    assert _prefix_rtc_timestep_bucket(_head("legacy_zero")) == 0


def test_prefix_rtc_groot_clean_uses_last_training_bucket():
    assert _prefix_rtc_timestep_bucket(_head("groot_clean")) == 999


def test_prefix_rtc_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported prefix_rtc_timestep_mode"):
        _prefix_rtc_timestep_bucket(_head("unknown"))
