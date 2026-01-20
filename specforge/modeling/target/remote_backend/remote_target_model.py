"""
Remote Eagle3 target model implementation.

This module provides RemoteEagle3TargetModel that implements the Eagle3TargetModel
interface by delegating inference to remote workers via ZeroMQ and retrieving
results from Mooncake Store.

The model supports both synchronous and asynchronous operation modes:
- Synchronous: `generate_eagle3_data()` blocks until result is ready
- Asynchronous: `submit_task()` returns immediately, `get_result()` retrieves result

Prefetching should be implemented in the training loop by submitting tasks
a few steps ahead of when results are needed.

To reduce network bandwidth, inference workers send last_hidden_states instead
of logits. The trainer computes logits locally using the lm_head weights.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from specforge.modeling.target.eagle3_target_model import (
    Eagle3TargetModel,
    Eagle3TargetOutput,
)
from specforge.modeling.target.target_head import TargetHead
from specforge.utils import padding

from .messages import InferenceTask, TaskNotification, TaskStatus, generate_task_id
from .mooncake_client import EagleMooncakeStore, MooncakeConfig
from .task_queue import (
    NotificationSubscriber,
    QueueConfig,
    TaskProducer,
)

logger = logging.getLogger(__name__)


@dataclass
class RemoteBackendConfig:
    """Configuration for the remote backend."""

    task_queue_addr: str = "tcp://localhost:5555"
    notify_addr: str = "tcp://localhost:5556"
    task_timeout: float = 300.0
    retry_count: int = 3
    retry_delay: float = 1.0
    dp_rank: int = 0
    dp_size: int = 1
    mooncake_config: MooncakeConfig = None
    queue_config: QueueConfig = None
    target_model_path: Optional[str] = None
    lm_head_key: str = "lm_head.weight"
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.mooncake_config is None:
            self.mooncake_config = MooncakeConfig()
        if self.queue_config is None:
            self.queue_config = QueueConfig(
                task_queue_addr=self.task_queue_addr,
                notify_addr=self.notify_addr,
            )
        if self.target_model_path is None:
            raise ValueError("target_model_path is required so we can compute logits locally!")


class RemoteEagle3TargetModel(Eagle3TargetModel):
    """
    Remote implementation of Eagle3TargetModel.
    
    This class sends inference tasks to remote workers via ZeroMQ and retrieves
    results from Mooncake Store. It implements the same interface as local
    backends (SGLang, HuggingFace) for seamless integration.
    """

    def __init__(
        self,
        config: RemoteBackendConfig,
        aux_hidden_states_layers: Optional[List[int]] = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.config = config
        self.aux_hidden_states_layers = aux_hidden_states_layers or [1, -1, -4]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.task_producer: Optional[TaskProducer] = None
        self.notification_sub: Optional[NotificationSubscriber] = None
        self.mooncake_store: Optional[EagleMooncakeStore] = None
        self.target_head: Optional[TargetHead] = None

        self._connected = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "RemoteEagle3TargetModel":
        """
        Create a RemoteEagle3TargetModel from configuration.
        
        Note: Unlike local backends, this doesn't load the full model locally.
        The model is loaded on remote inference workers. Only the lm_head is
        loaded locally to compute logits from last_hidden_states.
        """
        config = kwargs.get("remote_config")
        if config is None:
            config = RemoteBackendConfig(
                task_queue_addr=kwargs.get("task_queue_addr", "tcp://localhost:5555"),
                notify_addr=kwargs.get("notify_addr", "tcp://localhost:5556"),
                task_timeout=kwargs.get("task_timeout", 300.0),
                target_model_path=pretrained_model_name_or_path,
                lm_head_key=kwargs.get("lm_head_key", "lm_head.weight"),
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )

            mooncake_config = MooncakeConfig(
                local_hostname=kwargs.get("mooncake_local_hostname", "localhost"),
                metadata_server=kwargs.get(
                    "mooncake_metadata_server", "http://localhost:8090/metadata"
                ),
                master_server_address=kwargs.get(
                    "mooncake_master_addr", "localhost:50051"
                ),
                global_segment_size=kwargs.get(
                    "mooncake_global_segment_size", 4 * 1024 * 1024 * 1024
                ),
                local_buffer_size=kwargs.get(
                    "mooncake_local_buffer_size", 512 * 1024 * 1024
                ),
                protocol=kwargs.get("mooncake_protocol", "tcp"),
                device_name=kwargs.get("mooncake_device_name", ""),
            )
            config.mooncake_config = mooncake_config
        else:
            if config.target_model_path is None:
                config.target_model_path = pretrained_model_name_or_path

        aux_hidden_states_layers = kwargs.get("aux_hidden_states_layers")
        return cls(config=config, aux_hidden_states_layers=aux_hidden_states_layers)

    @classmethod
    def from_config(cls, config: RemoteBackendConfig) -> "RemoteEagle3TargetModel":
        """Create from a RemoteBackendConfig object."""
        return cls(config=config)

    def connect(self) -> None:
        """Connect to task queue and notification system."""
        if self._connected:
            return

        self.task_producer = TaskProducer(self.config.queue_config)
        self.task_producer.connect(self.config.task_queue_addr)

        self.notification_sub = NotificationSubscriber(self.config.queue_config)
        self.notification_sub.connect(self.config.notify_addr)
        self.notification_sub.start_listener()

        self.mooncake_store = EagleMooncakeStore(self.config.mooncake_config)
        self.mooncake_store.setup()

        if self.config.target_model_path is not None:
            self.target_head = TargetHead.from_pretrained(
                model_path=self.config.target_model_path,
                lm_head_key=self.config.lm_head_key,
                trust_remote_code=self.config.trust_remote_code,
            )
            logger.info(f"Loaded TargetHead from {self.config.target_model_path}")

        self._connected = True
        logger.info("RemoteEagle3TargetModel connected to remote infrastructure")

    def disconnect(self) -> None:
        """Disconnect from all remote services."""
        if self.task_producer is not None:
            self.task_producer.close()
            self.task_producer = None

        if self.notification_sub is not None:
            self.notification_sub.close()
            self.notification_sub = None

        if self.mooncake_store is not None:
            self.mooncake_store.close()
            self.mooncake_store = None

        if self.target_head is not None:
            self.target_head = None

        self._connected = False
        logger.info("RemoteEagle3TargetModel disconnected")

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """
        Set the layers to capture the aux hidden states from the target model outputs.
        """
        if aux_hidden_states_layers is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
            else:
                raise ValueError(
                    f"Failed to set aux hidden states layers as model config {self.model.config} does not have num_hidden_layers"
                )
            aux_hidden_states_layers = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers for EAGLE3"

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        """
        Generate Eagle3 data by delegating to remote inference workers.
        
        This method:
        1. Submits an inference task to the task queue
        2. Waits for completion notification
        3. Retrieves results from Mooncake Store (using zero-copy if enabled)
        4. Computes logits locally from last_hidden_states using TargetHead
        """
        if not self._connected:
            self.connect()

        task_id = generate_task_id(rank=self.config.dp_rank)

        task = InferenceTask.create(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states_layers=self.aux_hidden_states_layers,
            return_last_hidden_states=True,
            return_logits=False,
            task_id=task_id,
        )

        self.notification_sub.subscribe(task_id)

        try:
            self.task_producer.push(task)
            logger.debug(f"[Rank {self.config.dp_rank}] Submitted task {task_id}")

            notification = self.notification_sub.wait_for(
                task_id, timeout=self.config.task_timeout
            )

            if notification is None:
                raise TimeoutError(
                    f"Timeout waiting for task {task_id} after {self.config.task_timeout}s"
                )

            if notification.status == TaskStatus.FAILED:
                raise RuntimeError(
                    f"Remote inference failed for task {task_id}: {notification.error_message}"
                )

            if notification.status != TaskStatus.READY:
                raise RuntimeError(
                    f"Unexpected task status for {task_id}: {notification.status}"
                )

            output = self._retrieve_output(notification, input_ids.device)

            self._cleanup_mooncake_data(notification)

            logger.debug(f"Retrieved results for task {task_id}")
            return output

        finally:
            self.notification_sub.unsubscribe(task_id)

    def _retrieve_output(
        self,
        notification: TaskNotification,
        device: torch.device,
    ) -> Eagle3TargetOutput:
        """Retrieve output from Mooncake, using tensor API or legacy path.
        
        If target (logits) is not present but last_hidden_states is, compute
        logits locally using the TargetHead to save network bandwidth.
        """
        mooncake_key = notification.mooncake_key or notification.task_id

        if notification.use_tensor_api and notification.tensor_shapes is not None:
            dtypes = notification.tensor_dtypes or {}
            output = self.mooncake_store.get_eagle3_tensors_into(
                key=mooncake_key,
                shapes=notification.tensor_shapes,
                dtypes=dtypes,
                device=device,
            )
            logger.debug(f"Retrieved tensors from Mooncake: {mooncake_key}")

            if output.target is None and output.last_hidden_states is not None:
                if self.target_head is None:
                    raise RuntimeError(
                        "Received last_hidden_states without target, but TargetHead is not initialized. "
                        "Please ensure target_model_path is set in RemoteBackendConfig."
                    )
                if output.last_hidden_states.device == "cpu":
                    logger.info(f"Moving last_hidden_states to device {device}")
                    output.last_hidden_states = output.last_hidden_states.to(device)
                target = self.target_head(output.last_hidden_states)
                output.target = padding(target, left=False)
                logger.debug("Computed logits from last_hidden_states using TargetHead")

            return output
        else:
            return self.mooncake_store.get_eagle3_output(mooncake_key, device=device)

    def _cleanup_mooncake_data(self, notification: TaskNotification) -> None:
        """Remove data from Mooncake after retrieval."""
        mooncake_key = notification.mooncake_key or notification.task_id

        if notification.use_tensor_api:
            shapes = notification.tensor_shapes or {}
            has_lhs = "last_hidden_states" in shapes
            has_target = "target" in shapes
            self.mooncake_store.remove_eagle3_tensors(
                mooncake_key,
                has_last_hidden_states=has_lhs,
                has_target=has_target,
            )
        else:
            self.mooncake_store.remove(mooncake_key)

    def submit_task(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> str:
        """
        Submit an inference task asynchronously.
        
        Returns the task_id which can be used to retrieve the result later via get_result().
        This enables prefetching by submitting tasks ahead of when results are needed.
        """
        if not self._connected:
            self.connect()

        task_id = generate_task_id(rank=self.config.dp_rank)

        task = InferenceTask.create(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states_layers=self.aux_hidden_states_layers,
            return_last_hidden_states=True,
            return_logits=False,
            task_id=task_id,
        )

        self.notification_sub.subscribe(task_id)
        self.task_producer.push(task)
        logger.debug(f"[Rank {self.config.dp_rank}] Submitted task {task_id}")
        return task_id

    @torch.no_grad()
    def get_result(
        self,
        task_id: str,
        device: Optional[torch.device] = None,
        timeout: Optional[float] = None,
    ) -> Eagle3TargetOutput:
        """
        Get the result of a previously submitted task.
        
        Args:
            task_id: The task ID returned by submit_task()
            device: Device to place the output tensors on (default: cuda)
            timeout: Timeout in seconds (default: config.task_timeout)
            
        Returns:
            Eagle3TargetOutput with the inference results
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        device = device or self.device
        timeout = timeout or self.config.task_timeout

        try:
            notification = self.notification_sub.wait_for(task_id, timeout=timeout)

            if notification is None:
                raise TimeoutError(
                    f"Timeout waiting for task {task_id} after {timeout}s"
                )

            if notification.status == TaskStatus.FAILED:
                raise RuntimeError(
                    f"Remote inference failed for task {task_id}: {notification.error_message}"
                )

            if notification.status != TaskStatus.READY:
                raise RuntimeError(
                    f"Unexpected task status for {task_id}: {notification.status}"
                )

            mooncake_key = notification.mooncake_key or task_id
            output = self._retrieve_output(notification, device)
            self._cleanup_mooncake_data(notification)

            logger.debug(f"Retrieved results for task {task_id}")
            return output

        finally:
            self.notification_sub.unsubscribe(task_id)

    def generate_eagle3_data_batch(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        loss_mask_list: List[torch.Tensor],
    ) -> List[Eagle3TargetOutput]:
        """
        Generate Eagle3 data for multiple inputs in parallel.
        
        This method submits all tasks at once and waits for all results,
        which can be more efficient than sequential processing.
        """
        if not self._connected:
            self.connect()

        task_ids = []
        for input_ids, attention_mask, loss_mask in zip(
            input_ids_list, attention_mask_list, loss_mask_list
        ):
            task_id = self.submit_task(input_ids, attention_mask, loss_mask)
            task_ids.append(task_id)

        logger.debug(f"Submitted batch of {len(task_ids)} tasks")

        outputs = []
        for task_id in task_ids:
            output = self.get_result(task_id)
            outputs.append(output)

        return outputs

    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


if __name__ == "__main__":
    model = RemoteEagle3TargetModel.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen3-8B",
        remote_config=RemoteBackendConfig(
            task_queue_addr="tcp://localhost:5555",
            notify_addr="tcp://localhost:5556",
        ),
    )
    model.connect()
    model.generate_eagle3_data(
        input_ids=torch.randint(0, 100, (1, 1024)),
        attention_mask=torch.ones((1, 1024)),
        loss_mask=torch.ones((1, 1024)),
    )
    model.disconnect()