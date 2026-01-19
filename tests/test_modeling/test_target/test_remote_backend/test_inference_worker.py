"""Tests for the InferenceWorker class with mocked dependencies.

This module tests the InferenceWorker by:
1. Mocking the target model to return fake hidden states
2. Mocking the Mooncake store
3. Using real ZMQ task queue and notification system
4. Verifying the worker processes tasks correctly
"""

import random
import threading
import time
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from specforge.modeling.target.remote_backend.inference_worker import (
    InferenceWorker,
    InferenceWorkerConfig,
)
from specforge.modeling.target.remote_backend.messages import (
    InferenceTask,
    TaskNotification,
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

    def put_raw(self, key: str, data: bytes):
        self.data[key] = data

    def put_eagle3_output_data(self, key: str, output_data):
        self.data[key] = output_data.serialize()

    def get(self, key: str) -> Optional[bytes]:
        return self.data.get(key)

    def close(self):
        self.close_called = True


@pytest.fixture
def test_config():
    """Create configuration with unique ports for test isolation."""
    base_port = 17000 + random.randint(0, 1000)
    
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
    
    return worker_config, queue_config


class TestInferenceWorkerWithMocks:
    """Test InferenceWorker with mocked model and store."""

    def test_worker_processes_single_task(self, test_config):
        """Test that worker correctly processes a single task."""
        worker_config, queue_config = test_config
        
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
        
        def run_worker_briefly():
            start = time.time()
            worker._running = True
            while worker._running and (time.time() - start) < 5.0:
                task = worker.task_consumer.pull(timeout_ms=100)
                if task is not None:
                    worker._process_task(task)
                    break
            worker._running = False
        
        worker_thread = threading.Thread(target=run_worker_briefly, daemon=True)
        worker_thread.start()
        
        try:
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
            
            subscriber.subscribe(task.task_id)
            producer.push(task)
            
            notification = subscriber.wait_for(task.task_id, timeout=5.0)
            
            assert notification is not None, "Should receive notification"
            assert notification.status == TaskStatus.READY, f"Task should succeed, got {notification.status}"
            assert notification.mooncake_key == "test-worker-task-1"
            assert task.task_id in mock_store.data, "Output should be stored"
            assert mock_model.extend_call_count == 1, "Model should be called once"
            
        finally:
            worker._running = False
            worker_thread.join(timeout=2.0)
            producer.close()
            subscriber.close()
            worker.task_consumer.close()
            worker.notification_pub.close()

    def test_worker_processes_multiple_tasks(self, test_config):
        """Test that worker correctly processes multiple tasks."""
        worker_config, queue_config = test_config
        
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
        tasks_processed = []
        tasks_processed_lock = threading.Lock()
        
        def run_worker_for_tasks():
            start = time.time()
            worker._running = True
            processed = 0
            while worker._running and (time.time() - start) < 10.0 and processed < num_tasks:
                task = worker.task_consumer.pull(timeout_ms=100)
                if task is not None:
                    worker._process_task(task)
                    with tasks_processed_lock:
                        tasks_processed.append(task.task_id)
                    processed += 1
            worker._running = False
        
        worker_thread = threading.Thread(target=run_worker_for_tasks, daemon=True)
        worker_thread.start()
        
        try:
            task_ids = []
            for i in range(num_tasks):
                input_ids = torch.randint(0, 100, (1, 10 + i))
                attention_mask = torch.ones(1, 10 + i, dtype=torch.long)
                loss_mask = torch.ones(1, 10 + i, dtype=torch.long)
                
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
            
            completed = 0
            for task_id in task_ids:
                notification = subscriber.wait_for(task_id, timeout=5.0)
                if notification and notification.status == TaskStatus.READY:
                    completed += 1
            
            assert completed == num_tasks, f"All tasks should complete, got {completed}/{num_tasks}"
            assert mock_model.extend_call_count == num_tasks
            assert len(mock_store.data) == num_tasks
            
        finally:
            worker._running = False
            worker_thread.join(timeout=2.0)
            producer.close()
            subscriber.close()
            worker.task_consumer.close()
            worker.notification_pub.close()

    def test_worker_handles_different_batch_sizes(self, test_config):
        """Test worker handles different batch sizes correctly."""
        worker_config, queue_config = test_config
        
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
        
        def run_worker_briefly():
            start = time.time()
            worker._running = True
            processed = 0
            while worker._running and (time.time() - start) < 10.0 and processed < 2:
                task = worker.task_consumer.pull(timeout_ms=100)
                if task is not None:
                    worker._process_task(task)
                    processed += 1
            worker._running = False
        
        worker_thread = threading.Thread(target=run_worker_briefly, daemon=True)
        worker_thread.start()
        
        try:
            task1 = InferenceTask.create(
                input_ids=torch.randint(0, 100, (1, 20)),
                attention_mask=torch.ones(1, 20, dtype=torch.long),
                loss_mask=torch.ones(1, 20, dtype=torch.long),
                aux_hidden_states_layers=[1, -1, -4],
                task_id="batch-1-task",
            )
            
            task2 = InferenceTask.create(
                input_ids=torch.randint(0, 100, (4, 32)),
                attention_mask=torch.ones(4, 32, dtype=torch.long),
                loss_mask=torch.ones(4, 32, dtype=torch.long),
                aux_hidden_states_layers=[1, -1, -4],
                task_id="batch-4-task",
            )
            
            subscriber.subscribe(task1.task_id)
            subscriber.subscribe(task2.task_id)
            
            producer.push(task1)
            producer.push(task2)
            
            notif1 = subscriber.wait_for(task1.task_id, timeout=5.0)
            notif2 = subscriber.wait_for(task2.task_id, timeout=5.0)
            
            assert notif1 is not None and notif1.status == TaskStatus.READY
            assert notif2 is not None and notif2.status == TaskStatus.READY
            
        finally:
            worker._running = False
            worker_thread.join(timeout=2.0)
            producer.close()
            subscriber.close()
            worker.task_consumer.close()
            worker.notification_pub.close()

    def test_worker_respects_aux_hidden_states_layers(self, test_config):
        """Test that worker passes aux_hidden_states_layers to model."""
        worker_config, queue_config = test_config
        
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
        
        def run_worker_briefly():
            start = time.time()
            worker._running = True
            while worker._running and (time.time() - start) < 5.0:
                task = worker.task_consumer.pull(timeout_ms=100)
                if task is not None:
                    worker._process_task(task)
                    break
            worker._running = False
        
        worker_thread = threading.Thread(target=run_worker_briefly, daemon=True)
        worker_thread.start()
        
        try:
            custom_layers = [0, 2, -2, -1]
            task = InferenceTask.create(
                input_ids=torch.randint(0, 100, (1, 10)),
                attention_mask=torch.ones(1, 10, dtype=torch.long),
                loss_mask=torch.ones(1, 10, dtype=torch.long),
                aux_hidden_states_layers=custom_layers,
                task_id="custom-layers-task",
            )
            
            subscriber.subscribe(task.task_id)
            producer.push(task)
            
            notification = subscriber.wait_for(task.task_id, timeout=5.0)
            
            assert notification is not None
            assert notification.status == TaskStatus.READY
            assert mock_model.aux_hidden_states_layers == custom_layers
            
        finally:
            worker._running = False
            worker_thread.join(timeout=2.0)
            producer.close()
            subscriber.close()
            worker.task_consumer.close()
            worker.notification_pub.close()


class TestInferenceWorkerErrorHandling:
    """Test error handling in InferenceWorker."""

    def test_worker_publishes_failure_on_model_error(self, test_config):
        """Test that worker publishes FAILED notification when model raises error."""
        worker_config, queue_config = test_config
        
        mock_model = MockSGLangEagle3TargetModel()
        mock_store = MockEagleMooncakeStore()
        
        def raise_error(*args, **kwargs):
            raise RuntimeError("Simulated model error")
        mock_model.extend = raise_error
        
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
        
        def run_worker_briefly():
            start = time.time()
            worker._running = True
            while worker._running and (time.time() - start) < 5.0:
                task = worker.task_consumer.pull(timeout_ms=100)
                if task is not None:
                    worker._process_task(task)
                    break
            worker._running = False
        
        worker_thread = threading.Thread(target=run_worker_briefly, daemon=True)
        worker_thread.start()
        
        try:
            task = InferenceTask.create(
                input_ids=torch.randint(0, 100, (1, 10)),
                attention_mask=torch.ones(1, 10, dtype=torch.long),
                loss_mask=torch.ones(1, 10, dtype=torch.long),
                aux_hidden_states_layers=[1, -1],
                task_id="error-task",
            )
            
            subscriber.subscribe(task.task_id)
            producer.push(task)
            
            notification = subscriber.wait_for(task.task_id, timeout=5.0)
            
            assert notification is not None
            assert notification.status == TaskStatus.FAILED
            assert "Simulated model error" in notification.error_message
            
        finally:
            worker._running = False
            worker_thread.join(timeout=2.0)
            producer.close()
            subscriber.close()
            worker.task_consumer.close()
            worker.notification_pub.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
