"""Tests for message serialization and deserialization."""

import pytest
import torch

from specforge.modeling.target.remote_backend.messages import (
    Eagle3OutputData,
    InferenceTask,
    TaskNotification,
    TaskStatus,
    deserialize_tensor,
    generate_task_id,
    serialize_tensor,
)


class TestTensorSerialization:
    """Test tensor serialization/deserialization."""

    def test_serialize_deserialize_float_tensor(self):
        tensor = torch.randn(2, 4, 8)
        serialized = serialize_tensor(tensor)
        deserialized = deserialize_tensor(serialized)

        assert torch.allclose(tensor, deserialized)
        assert tensor.shape == deserialized.shape
        assert tensor.dtype == deserialized.dtype

    def test_serialize_deserialize_int_tensor(self):
        tensor = torch.randint(0, 100, (2, 4))
        serialized = serialize_tensor(tensor)
        deserialized = deserialize_tensor(serialized)

        assert torch.equal(tensor, deserialized)

    def test_serialize_deserialize_bfloat16_tensor(self):
        tensor = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        serialized = serialize_tensor(tensor)
        deserialized = deserialize_tensor(serialized)

        assert torch.allclose(tensor.float(), deserialized.float())
        assert tensor.dtype == deserialized.dtype


class TestInferenceTask:
    """Test InferenceTask message."""

    def test_create_from_tensors(self):
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        loss_mask = torch.ones(1, 10)

        task = InferenceTask.create(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )

        assert task.task_id is not None
        assert len(task.task_id) > 0
        assert task.aux_hidden_states_layers == [1, -1, -4]

    def test_create_with_custom_task_id(self):
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        loss_mask = torch.ones(1, 10)

        task = InferenceTask.create(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            task_id="custom-task-123",
        )

        assert task.task_id == "custom-task-123"

    def test_serialize_deserialize(self):
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        loss_mask = torch.ones(1, 10)

        task = InferenceTask.create(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states_layers=[1, 16, 28],
            return_last_hidden_states=True,
        )

        serialized = task.serialize()
        deserialized = InferenceTask.deserialize(serialized)

        assert deserialized.task_id == task.task_id
        assert deserialized.aux_hidden_states_layers == [1, 16, 28]
        assert deserialized.return_last_hidden_states is True

        assert torch.equal(
            task.get_input_ids(), deserialized.get_input_ids()
        )

    def test_get_tensors(self):
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        loss_mask = torch.zeros(1, 10)

        task = InferenceTask.create(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )

        assert torch.equal(task.get_input_ids(), input_ids)
        assert torch.equal(task.get_attention_mask(), attention_mask)
        assert torch.equal(task.get_loss_mask(), loss_mask)


class TestTaskNotification:
    """Test TaskNotification message."""

    def test_ready_notification(self):
        notification = TaskNotification(
            task_id="task-123",
            status=TaskStatus.READY,
            mooncake_key="task-123",
        )

        serialized = notification.serialize()
        deserialized = TaskNotification.deserialize(serialized)

        assert deserialized.task_id == "task-123"
        assert deserialized.status == TaskStatus.READY
        assert deserialized.mooncake_key == "task-123"
        assert deserialized.error_message is None

    def test_failed_notification(self):
        notification = TaskNotification(
            task_id="task-456",
            status=TaskStatus.FAILED,
            error_message="Model OOM error",
        )

        serialized = notification.serialize()
        deserialized = TaskNotification.deserialize(serialized)

        assert deserialized.status == TaskStatus.FAILED
        assert deserialized.error_message == "Model OOM error"


class TestEagle3OutputData:
    """Test Eagle3OutputData message."""

    def test_from_tensors(self):
        hidden_states = torch.randn(1, 10, 4096 * 3, dtype=torch.bfloat16)
        target = torch.randn(1, 10, 32000, dtype=torch.bfloat16)
        loss_mask = torch.ones(1, 10, 1)
        input_ids = torch.randint(0, 32000, (1, 10))
        attention_mask = torch.ones(1, 10)

        output_data = Eagle3OutputData.from_tensors(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert output_data.hidden_states_shape == (1, 10, 4096 * 3)
        assert output_data.target_shape == (1, 10, 32000)

    def test_serialize_deserialize(self):
        hidden_states = torch.randn(1, 5, 128, dtype=torch.bfloat16)
        target = torch.randn(1, 5, 100, dtype=torch.bfloat16)
        loss_mask = torch.ones(1, 5, 1)
        input_ids = torch.randint(0, 100, (1, 5))
        attention_mask = torch.ones(1, 5)

        output_data = Eagle3OutputData.from_tensors(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        serialized = output_data.serialize()
        deserialized = Eagle3OutputData.deserialize(serialized)

        assert deserialized.hidden_states_shape == output_data.hidden_states_shape
        assert deserialized.target_shape == output_data.target_shape

    def test_to_eagle3_output(self):
        hidden_states = torch.randn(1, 5, 128, dtype=torch.bfloat16)
        target = torch.randn(1, 5, 100, dtype=torch.bfloat16)
        loss_mask = torch.ones(1, 5, 1)
        input_ids = torch.randint(0, 100, (1, 5))
        attention_mask = torch.ones(1, 5)

        output_data = Eagle3OutputData.from_tensors(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        eagle3_output = output_data.to_eagle3_output(device="cpu")

        assert eagle3_output.hidden_states.shape == hidden_states.shape
        assert eagle3_output.target.shape == target.shape
        assert torch.equal(eagle3_output.input_ids, input_ids)


class TestGenerateTaskId:
    """Test task ID generation."""

    def test_unique_ids(self):
        ids = [generate_task_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_id_format(self):
        task_id = generate_task_id()
        assert isinstance(task_id, str)
        assert len(task_id) == 36
