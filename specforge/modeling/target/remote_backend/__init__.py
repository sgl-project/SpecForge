"""
Remote backend module for distributed inference with Mooncake Store.

This module provides components for:
- Distributed task queue (ZeroMQ-based)
- Mooncake Store integration for hidden state storage
- Remote inference worker service
- Training-side remote target model client
"""

from .messages import (
    Eagle3OutputData,
    InferenceTask,
    TaskNotification,
    TaskStatus,
    generate_task_id,
)
from .task_queue import (
    NotificationBroker,
    NotificationPublisher,
    NotificationSubscriber,
    QueueConfig,
    TaskConsumer,
    TaskProducer,
    TaskQueueBroker,
)
from .mooncake_client import (
    EagleMooncakeStore,
    MooncakeConfig,
    MooncakeHiddenStateStore,
    MooncakeHiddenStateStorePool,
)
from .remote_target_model import (
    RemoteBackendConfig,
    RemoteEagle3TargetModel,
)
from .inference_worker import (
    InferenceWorker,
    InferenceWorkerConfig,
    InferenceWorkerManager,
    run_worker,
)
from .zero_copy import (
    Eagle3BufferLayout,
    Eagle3ZeroCopyReader,
    Eagle3ZeroCopyWriter,
    GPUBuffer,
    GPUBufferPool,
    estimate_buffer_size,
)

__all__ = [
    "InferenceTask",
    "TaskNotification",
    "TaskStatus",
    "Eagle3OutputData",
    "generate_task_id",
    "QueueConfig",
    "TaskProducer",
    "TaskConsumer",
    "NotificationPublisher",
    "NotificationSubscriber",
    "TaskQueueBroker",
    "NotificationBroker",
    "MooncakeConfig",
    "MooncakeHiddenStateStore",
    "MooncakeHiddenStateStorePool",
    "EagleMooncakeStore",
    "RemoteBackendConfig",
    "RemoteEagle3TargetModel",
    "InferenceWorker",
    "InferenceWorkerConfig",
    "InferenceWorkerManager",
    "run_worker",
    "GPUBuffer",
    "GPUBufferPool",
    "Eagle3BufferLayout",
    "Eagle3ZeroCopyWriter",
    "Eagle3ZeroCopyReader",
    "estimate_buffer_size",
]
