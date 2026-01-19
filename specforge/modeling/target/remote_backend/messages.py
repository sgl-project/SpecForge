"""
Message definitions for distributed inference communication.

This module defines the data structures used for communication between
training nodes and inference workers via ZeroMQ and Mooncake Store.
"""

import io
import pickle
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch


class TaskStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


@dataclass
class InferenceTask:
    """
    TODO: Delay the tokenization to the inference worker and send metadata only.
    or simply leverage transfer tensor.
    """

    task_id: str
    input_ids: bytes
    attention_mask: bytes
    loss_mask: bytes
    aux_hidden_states_layers: List[int] = field(default_factory=lambda: [1, -1, -4])
    return_last_hidden_states: bool = False
    return_logits: bool = True

    @classmethod
    def create(
        cls,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        aux_hidden_states_layers: List[int],
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        task_id: Optional[str] = None,
    ) -> "InferenceTask":
        if task_id is None:
            task_id = str(uuid.uuid4())
        return cls(
            task_id=task_id,
            input_ids=serialize_tensor(input_ids),
            attention_mask=serialize_tensor(attention_mask),
            loss_mask=serialize_tensor(loss_mask),
            aux_hidden_states_layers=aux_hidden_states_layers,
            return_last_hidden_states=return_last_hidden_states,
            return_logits=return_logits,
        )

    def get_input_ids(self) -> torch.Tensor:
        return deserialize_tensor(self.input_ids)

    def get_attention_mask(self) -> torch.Tensor:
        return deserialize_tensor(self.attention_mask)

    def get_loss_mask(self) -> torch.Tensor:
        return deserialize_tensor(self.loss_mask)

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes) -> "InferenceTask":
        return pickle.loads(data)


@dataclass
class TaskNotification:
    """Notification sent by inference workers when a task is completed."""

    task_id: str
    status: TaskStatus
    mooncake_key: Optional[str] = None
    error_message: Optional[str] = None
    tensor_shapes: Optional[dict] = None
    data_size: int = 0

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes) -> "TaskNotification":
        return pickle.loads(data)

def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a PyTorch tensor to bytes."""
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserialize bytes back to a PyTorch tensor."""
    buffer = io.BytesIO(data)
    return torch.load(buffer, weights_only=True)


def generate_task_id(rank: Optional[int] = None) -> str:
    """
    Generate a unique task ID.
    
    Args:
        rank: Optional DP rank to include in the ID for easier debugging
              and to ensure uniqueness across ranks.
    """
    uid = str(uuid.uuid4())
    if rank is not None:
        return f"r{rank}_{uid}"
    return uid


@dataclass
class Eagle3OutputData:
    """
    Serialized Eagle3 output data for storage in Mooncake Store.
    
    Stores tensors as bytes to avoid pickling issues across processes.
    """

    hidden_states: bytes
    target: Optional[bytes]
    loss_mask: bytes
    input_ids: bytes
    attention_mask: bytes
    last_hidden_states: Optional[bytes] = None

    @classmethod
    def from_tensors(
        cls,
        hidden_states: torch.Tensor,
        target: Optional[torch.Tensor],
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> "Eagle3OutputData":
        """Create from tensors by serializing them."""
        return cls(
            hidden_states=serialize_tensor(hidden_states),
            target=serialize_tensor(target) if target is not None else None,
            loss_mask=serialize_tensor(loss_mask),
            input_ids=serialize_tensor(input_ids),
            attention_mask=serialize_tensor(attention_mask),
            last_hidden_states=(
                serialize_tensor(last_hidden_states)
                if last_hidden_states is not None
                else None
            ),
        )

    def to_eagle3_output(self, device: str = "cuda"):
        """Convert to Eagle3TargetOutput with tensors on the specified device."""
        from ..eagle3_target_model import Eagle3TargetOutput

        return Eagle3TargetOutput(
            hidden_states=deserialize_tensor(self.hidden_states).to(device),
            target=(
                deserialize_tensor(self.target).to(device)
                if self.target is not None
                else None
            ),
            loss_mask=deserialize_tensor(self.loss_mask).to(device),
            input_ids=deserialize_tensor(self.input_ids).to(device),
            attention_mask=deserialize_tensor(self.attention_mask).to(device),
            last_hidden_states=(
                deserialize_tensor(self.last_hidden_states).to(device)
                if self.last_hidden_states is not None
                else None
            ),
        )

    def serialize(self) -> bytes:
        """Serialize the entire object."""
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes) -> "Eagle3OutputData":
        """Deserialize from bytes."""
        return pickle.loads(data)


if __name__ == "__main__":
    task = InferenceTask.create(
        input_ids=torch.randint(0, 100, (16, 1000)),
        attention_mask=torch.randint(0, 1, (16, 1000)),
        loss_mask=torch.randint(0, 1, (16, 1000)),
        aux_hidden_states_layers=[1, -1, -4],
        return_last_hidden_states=False,
        return_logits=True,
    )
    print(task.get_input_ids())