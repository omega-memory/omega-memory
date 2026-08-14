"""Configuration contract for content-dedup thresholds."""

import pytest


def test_default_thresholds_are_preserved_without_override():
    from omega.dedup_config import DEFAULT_DEDUP_THRESHOLDS, load_dedup_thresholds

    loaded = load_dedup_thresholds(env={})
    assert loaded == DEFAULT_DEDUP_THRESHOLDS
    assert loaded is not DEFAULT_DEDUP_THRESHOLDS


def test_selected_thresholds_can_be_overridden_without_mutating_defaults():
    from omega.dedup_config import DEFAULT_DEDUP_THRESHOLDS, load_dedup_thresholds

    original = dict(DEFAULT_DEDUP_THRESHOLDS)
    loaded = load_dedup_thresholds(
        env={"OMEGA_DEDUP_THRESHOLDS": '{"memory":0.92,"user_fact":0.90}'}
    )
    assert loaded["memory"] == 0.92
    assert loaded["user_fact"] == 0.90
    assert DEFAULT_DEDUP_THRESHOLDS == original


@pytest.mark.parametrize(
    ("payload", "bad_key"),
    [
        ('{"unknown": 0.5}', "unknown"),
        ('{"memory": true}', "memory"),
        ('{"memory": "high"}', "memory"),
        ('{"memory": -0.01}', "memory"),
        ('{"memory": 1.01}', "memory"),
    ],
)
def test_invalid_thresholds_name_the_bad_key(payload, bad_key):
    from omega.dedup_config import load_dedup_thresholds

    with pytest.raises(ValueError, match=bad_key):
        load_dedup_thresholds(env={"OMEGA_DEDUP_THRESHOLDS": payload})


def test_override_must_be_a_json_object():
    from omega.dedup_config import load_dedup_thresholds

    with pytest.raises(ValueError, match="object"):
        load_dedup_thresholds(env={"OMEGA_DEDUP_THRESHOLDS": "[]"})
