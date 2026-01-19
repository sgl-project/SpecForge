"""Tests for Mooncake Store client wrapper."""

import pytest
import torch

from specforge.modeling.target.remote_backend.mooncake_client import (
    EagleMooncakeStore,
    MooncakeConfig,
)


class TestMooncakeConfig:
    """Tests for MooncakeConfig."""

    def test_default_config(self):
        config = MooncakeConfig()
        assert config.local_hostname == "localhost"
        assert config.protocol == "tcp"
        assert config.global_segment_size == 4 * 1024 * 1024 * 1024

    def test_parse_size(self):
        assert MooncakeConfig.parse_size("4GB") == 4 * 1024 * 1024 * 1024
        assert MooncakeConfig.parse_size("512MB") == 512 * 1024 * 1024
        assert MooncakeConfig.parse_size("1024KB") == 1024 * 1024
        assert MooncakeConfig.parse_size("1024") == 1024

    def test_parse_size_case_insensitive(self):
        assert MooncakeConfig.parse_size("4gb") == 4 * 1024 * 1024 * 1024
        assert MooncakeConfig.parse_size("512Mb") == 512 * 1024 * 1024

    def test_parse_size_with_decimals(self):
        assert MooncakeConfig.parse_size("1.5GB") == int(1.5 * 1024 * 1024 * 1024)
        assert MooncakeConfig.parse_size("0.5GB") == int(0.5 * 1024 * 1024 * 1024)


class TestEagleMooncakeStoreFallback:
    """Tests for EagleMooncakeStore using fallback in-memory storage."""

    @pytest.fixture
    def store(self):
        config = MooncakeConfig()
        store = EagleMooncakeStore(config)
        store.setup()
        yield store
        store.close()

    def test_put_and_get(self, store):
        """Test basic put and get operations."""
        hidden_states = torch.randn(1, 10, 128, dtype=torch.bfloat16)
        target = torch.randn(1, 10, 100, dtype=torch.bfloat16)
        loss_mask = torch.ones(1, 10, 1)
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10)

        store.put_eagle3_output(
            key="test-key-1",
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        output = store.get_eagle3_output("test-key-1", device="cpu")

        assert output.hidden_states.shape == hidden_states.shape
        assert output.target.shape == target.shape
        assert torch.equal(output.input_ids, input_ids)

    def test_get_nonexistent_key(self, store):
        """Test that getting nonexistent key raises KeyError."""
        with pytest.raises(KeyError):
            store.get_eagle3_output("nonexistent-key")

    def test_remove(self, store):
        """Test removing a key."""
        hidden_states = torch.randn(1, 10, 128)
        target = torch.randn(1, 10, 100)
        loss_mask = torch.ones(1, 10, 1)
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10)

        store.put_eagle3_output(
            key="to-remove",
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        store.remove("to-remove")

        with pytest.raises(KeyError):
            store.get_eagle3_output("to-remove")

    def test_exists(self, store):
        """Test checking if key exists."""
        hidden_states = torch.randn(1, 10, 128)
        target = torch.randn(1, 10, 100)
        loss_mask = torch.ones(1, 10, 1)
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10)

        assert not store.exists("check-key")

        store.put_eagle3_output(
            key="check-key",
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert store.exists("check-key")

    def test_multiple_keys(self, store):
        """Test storing and retrieving multiple keys."""
        for i in range(5):
            hidden_states = torch.randn(1, 10, 128)
            target = torch.randn(1, 10, 100)
            loss_mask = torch.ones(1, 10, 1)
            input_ids = torch.randint(0, 100, (1, 10))
            attention_mask = torch.ones(1, 10)

            store.put_eagle3_output(
                key=f"multi-key-{i}",
                hidden_states=hidden_states,
                target=target,
                loss_mask=loss_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        for i in range(5):
            output = store.get_eagle3_output(f"multi-key-{i}", device="cpu")
            assert output is not None

    def test_with_last_hidden_states(self, store):
        """Test storing with last_hidden_states."""
        hidden_states = torch.randn(1, 10, 128)
        target = torch.randn(1, 10, 100)
        loss_mask = torch.ones(1, 10, 1)
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10)
        last_hidden_states = torch.randn(1, 10, 256)

        store.put_eagle3_output(
            key="with-last-hidden",
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states,
        )

        output = store.get_eagle3_output("with-last-hidden", device="cpu")
        assert output.last_hidden_states is not None
        assert output.last_hidden_states.shape == last_hidden_states.shape
