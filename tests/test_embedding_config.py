"""Tests for OMEGA_EMBEDDING_* configuration.

Multilingual deployments run non-default models (bge-m3: 1024-dim, CLS
pooling). Before this module the model name, directory, pooling, pad token and
dimension were literals in five files, so every upgrade reverted them and the
resulting dimension mismatch lost vectors silently.

These tests pin the contract: values come from the environment, and malformed
values raise at load time rather than reaching the store.
"""

import pytest

from omega.embedding_config import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    load_embedding_config,
)


def test_defaults_match_shipped_model():
    """No configuration means the factory bge-small-en-v1.5 setup."""
    config = load_embedding_config(env={})
    assert config.model_name == DEFAULT_EMBEDDING_MODEL
    assert config.dim == DEFAULT_EMBEDDING_DIM
    assert config.pooling == "mean"
    assert config.model_dir.endswith("bge-small-en-v1.5-onnx")


def test_bge_m3_configuration_round_trips():
    """The multilingual case this feature exists for."""
    config = load_embedding_config(
        env={
            "OMEGA_EMBEDDING_MODEL": "bge-m3",
            "OMEGA_EMBEDDING_DIM": "1024",
            "OMEGA_EMBEDDING_POOLING": "cls",
        }
    )
    assert config.model_name == "bge-m3"
    assert config.dim == 1024
    assert config.pooling == "cls"
    # Model dir follows the model name unless overridden.
    assert config.model_dir.endswith("bge-m3-onnx")


def test_explicit_model_dir_overrides_derived_path():
    config = load_embedding_config(
        env={"OMEGA_EMBEDDING_MODEL": "bge-m3", "OMEGA_EMBEDDING_MODEL_DIR": "/models/custom"}
    )
    assert config.model_dir == "/models/custom"


def test_pad_token_and_id_are_configurable():
    config = load_embedding_config(
        env={"OMEGA_EMBEDDING_PAD_TOKEN": "<pad>", "OMEGA_EMBEDDING_PAD_ID": "1"}
    )
    assert config.pad_token == "<pad>"
    assert config.pad_id == 1


@pytest.mark.parametrize(
    "env",
    [
        {"OMEGA_EMBEDDING_DIM": "not-a-number"},
        {"OMEGA_EMBEDDING_DIM": "0"},
        {"OMEGA_EMBEDDING_DIM": "-1"},
        {"OMEGA_EMBEDDING_DIM": "999999"},
        {"OMEGA_EMBEDDING_POOLING": "max"},
        {"OMEGA_EMBEDDING_PAD_ID": "-1"},
        {"OMEGA_EMBEDDING_PAD_ID": "not-a-number"},
    ],
)
def test_malformed_values_raise(env):
    """Fail at load, not at the first write."""
    with pytest.raises(ValueError):
        load_embedding_config(env=env)


def test_unset_and_blank_values_fall_back_to_defaults():
    """A blank var is 'unset', not an error — matches shell export habits."""
    config = load_embedding_config(env={"OMEGA_EMBEDDING_DIM": "  ", "OMEGA_EMBEDDING_POOLING": ""})
    assert config.dim == DEFAULT_EMBEDDING_DIM
    assert config.pooling == "mean"
