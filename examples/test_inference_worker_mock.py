#!/usr/bin/env python3
"""
Standalone test script for InferenceWorker with mocked dependencies.

This script tests the InferenceWorker by:
1. Mocking the target model to return fake hidden states
2. Mocking the Mooncake store
3. Using real ZMQ task queue and notification system
4. Sending a fake task and verifying the worker processes it

Usage:
    python examples/test_inference_worker_mock.py
"""

import random
import threading
import time
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock

import torch

from specforge.modeling.target.remote_backend.inference_worker import (
    InferenceWorker,
    InferenceWorkerConfig,
)
from specforge.modeling.target.remote_backend.messages import (
    InferenceTask,
    TaskStatus,
)
from specforge.modeling.target.remote_backend.mooncake_client import MooncakeConfig
from specforge.modeling.target.remote_backend.task_queue import (
    NotificationPublisher,
    NotificationSubscriber,
    QueueConfig,
    TaskConsumer,
    TaskProducer,
)


@dataclass
class MockEagle3TargetOutput:
    """Mock output from the target model."""
    hidden_states: torch.Tensor
    target: Optional[torch.Tensor]
    loss_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_states: Optional[torch.Tensor] = None


class MockSGLangEagle3TargetModel:
    """Mock target model that returns fake hidden states."""

    def __init__(self, hidden_dim: int = 128, vocab_size: int = 100):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.aux_hidden_states_layers: List[int] = [1, -1, -4]
        self.extend_call_count = 0

    def set_aux_hidden_states_layers(self, layers: Optional[List[int]]) -> None:
        if layers is not None:
            self.aux_hidden_states_layers = layers

    def extend(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
    ):
        """Mock extend method that returns fake data."""
        self.extend_call_count += 1

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        num_layers = len(self.aux_hidden_states_layers)

        data_cache = []
        logits_list = []
        aux_hidden_states_list = []
        last_hidden_states_list = []

        for i in range(batch_size):
            data_cache.append((
                input_ids[i:i+1],
                attention_mask[i:i+1],
                loss_mask[i:i+1],
            ))

            if return_logits:
                logits_list.append(
                    torch.randn(seq_len, self.vocab_size, dtype=torch.bfloat16)
                )
            else:
                logits_list.append(None)

            aux_hidden_states_list.append(
                torch.randn(num_layers, seq_len, self.hidden_dim, dtype=torch.bfloat16)
            )

            if return_last_hidden_states:
                last_hidden_states_list.append(
                    torch.randn(seq_len, self.hidden_dim, dtype=torch.bfloat16)
                )
            else:
                last_hidden_states_list.append(None)

        return data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()


class MockEagleMooncakeStore:
    """Mock Mooncake store for testing."""

    def __init__(self):
        self.data = {}
        self.setup_called = False
        self.close_called = False

    def setup(self):
        self.setup_called = True
        print("[MockStore] Setup complete")

    def put_raw(self, key: str, data: bytes):
        self.data[key] = data
        print(f"[MockStore] Stored {len(data)} bytes with key: {key}")

    def put_eagle3_output_data(self, key: str, output_data):
        self.data[key] = output_data.serialize()
        print(f"[MockStore] Stored output data with key: {key}")

    def get(self, key: str) -> Optional[bytes]:
        return self.data.get(key)

    def close(self):
        self.close_called = True
        print("[MockStore] Closed")


def run_test():
    """Run the inference worker test."""
    print("=" * 60)
    print("Testing InferenceWorker with Mocked Dependencies")
    print("=" * 60)

    base_port = 18000 + random.randint(0, 1000)
    print(f"\nUsing ports: {base_port} (task queue), {base_port + 1} (notifications)")

    queue_config = QueueConfig(
        task_queue_addr=f"tcp://127.0.0.1:{base_port}",
        notify_addr=f"tcp://127.0.0.1:{base_port + 1}",
    )

    mooncake_config = MooncakeConfig()

    worker_config = InferenceWorkerConfig(
        model_path="mock-model",
        task_queue_addr=queue_config.task_queue_addr,
        notify_addr=queue_config.notify_addr,
        worker_id=0,
        use_zero_copy=False,
        mooncake_config=mooncake_config,
        queue_config=queue_config,
    )

    print("\n[1] Creating mock components...")
    mock_model = MockSGLangEagle3TargetModel()
    mock_store = MockEagleMooncakeStore()

    print("[2] Creating InferenceWorker with mocked dependencies...")
    worker = InferenceWorker(worker_config)

    worker._setup_target_model = MagicMock()
    worker._setup_mooncake = MagicMock()
    worker.target_model = mock_model
    worker.mooncake_store = mock_store

    worker.task_consumer = TaskConsumer(queue_config)
    worker.task_consumer.bind()
    
    worker.notification_pub = NotificationPublisher(queue_config)
    worker.notification_pub.bind()
    
    mock_store.setup()

    print("[3] Creating TaskProducer and NotificationSubscriber...")
    producer = TaskProducer(queue_config)
    producer.connect()

    subscriber = NotificationSubscriber(queue_config)
    subscriber.connect()
    subscriber.start_listener()

    time.sleep(0.2)

    def run_worker_briefly():
        start = time.time()
        worker._running = True
        print("[Worker] Started processing loop")
        while worker._running and (time.time() - start) < 10.0:
            task = worker.task_consumer.pull(timeout_ms=100)
            if task is not None:
                print(f"[Worker] Received task: {task.task_id}")
                worker._process_task(task)
                print(f"[Worker] Finished processing task: {task.task_id}")
                break
        worker._running = False
        print("[Worker] Stopped")

    print("[4] Starting worker in background thread...")
    worker_thread = threading.Thread(target=run_worker_briefly, daemon=True)
    worker_thread.start()

    try:
        print("\n[5] Creating and sending test task...")
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        loss_mask = torch.ones(2, 16, dtype=torch.long)

        task = InferenceTask.create(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states_layers=[1, -1, -4],
            task_id="test-worker-task-1",
        )

        print(f"    Task ID: {task.task_id}")
        print(f"    Input shape: {input_ids.shape}")

        subscriber.subscribe(task.task_id)
        producer.push(task)
        print("    Task sent!")

        print("\n[6] Waiting for notification...")
        notification = subscriber.wait_for(task.task_id, timeout=10.0)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        if notification is None:
            print("❌ FAILED: No notification received (timeout)")
            return False

        print(f"✓ Notification received")
        print(f"    Status: {notification.status}")
        print(f"    Mooncake key: {notification.mooncake_key}")

        if notification.status == TaskStatus.READY:
            print(f"✓ Task completed successfully")
        else:
            print(f"❌ Task failed: {notification.error_message}")
            return False

        if task.task_id in mock_store.data:
            print(f"✓ Output stored in Mooncake (size: {len(mock_store.data[task.task_id])} bytes)")
        else:
            print("❌ Output not found in store")
            return False

        print(f"✓ Model called {mock_model.extend_call_count} time(s)")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True

    finally:
        worker._running = False
        worker_thread.join(timeout=2.0)
        producer.close()
        subscriber.close()
        worker.task_consumer.close()
        worker.notification_pub.close()


def run_multiple_tasks_test():
    """Test processing multiple tasks."""
    print("\n" + "=" * 60)
    print("Testing Multiple Tasks")
    print("=" * 60)

    base_port = 19000 + random.randint(0, 1000)

    queue_config = QueueConfig(
        task_queue_addr=f"tcp://127.0.0.1:{base_port}",
        notify_addr=f"tcp://127.0.0.1:{base_port + 1}",
    )

    mooncake_config = MooncakeConfig()

    worker_config = InferenceWorkerConfig(
        model_path="mock-model",
        task_queue_addr=queue_config.task_queue_addr,
        notify_addr=queue_config.notify_addr,
        worker_id=0,
        use_zero_copy=False,
        mooncake_config=mooncake_config,
        queue_config=queue_config,
    )

    mock_model = MockSGLangEagle3TargetModel()
    mock_store = MockEagleMooncakeStore()

    worker = InferenceWorker(worker_config)
    worker._setup_target_model = MagicMock()
    worker._setup_mooncake = MagicMock()
    worker.target_model = mock_model
    worker.mooncake_store = mock_store

    worker.task_consumer = TaskConsumer(queue_config)
    worker.task_consumer.bind()
    
    worker.notification_pub = NotificationPublisher(queue_config)
    worker.notification_pub.bind()
    
    mock_store.setup()

    producer = TaskProducer(queue_config)
    producer.connect()

    subscriber = NotificationSubscriber(queue_config)
    subscriber.connect()
    subscriber.start_listener()

    time.sleep(0.2)

    num_tasks = 3

    def run_worker_for_tasks():
        start = time.time()
        worker._running = True
        processed = 0
        while worker._running and (time.time() - start) < 15.0 and processed < num_tasks:
            task = worker.task_consumer.pull(timeout_ms=100)
            if task is not None:
                print(f"[Worker] Processing task {processed + 1}/{num_tasks}: {task.task_id}")
                worker._process_task(task)
                processed += 1
        worker._running = False

    worker_thread = threading.Thread(target=run_worker_for_tasks, daemon=True)
    worker_thread.start()

    try:
        task_ids = []
        for i in range(num_tasks):
            input_ids = torch.randint(0, 100, (1, 10 + i * 5))
            attention_mask = torch.ones(1, 10 + i * 5, dtype=torch.long)
            loss_mask = torch.ones(1, 10 + i * 5, dtype=torch.long)

            task = InferenceTask.create(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                aux_hidden_states_layers=[1, -1],
                task_id=f"multi-task-{i}",
            )
            task_ids.append(task.task_id)
            subscriber.subscribe(task.task_id)
            producer.push(task)
            print(f"Sent task {i + 1}/{num_tasks}: {task.task_id} (seq_len={10 + i * 5})")

        completed = 0
        for task_id in task_ids:
            notification = subscriber.wait_for(task_id, timeout=10.0)
            if notification and notification.status == TaskStatus.READY:
                completed += 1
                print(f"✓ Task {task_id} completed")

        print(f"\nCompleted {completed}/{num_tasks} tasks")
        print(f"Model called {mock_model.extend_call_count} times")

        if completed == num_tasks:
            print("✓ ALL MULTIPLE TASKS PASSED!")
            return True
        else:
            print("❌ Some tasks failed")
            return False

    finally:
        worker._running = False
        worker_thread.join(timeout=2.0)
        producer.close()
        subscriber.close()
        worker.task_consumer.close()
        worker.notification_pub.close()


if __name__ == "__main__":
    success1 = run_test()
    success2 = run_multiple_tasks_test()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
