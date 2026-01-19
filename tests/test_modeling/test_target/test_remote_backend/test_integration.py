"""Integration tests for the distributed inference pipeline.

These tests verify that all components work together correctly:
- Task queue (producer/consumer)
- Notification system (publisher/subscriber)  
- Mooncake Store (hidden state storage)
- RemoteEagle3TargetModel (client interface)
"""

import threading
import time

import pytest
import torch

from specforge.modeling.target.remote_backend.messages import (
    Eagle3OutputData,
    InferenceTask,
    TaskNotification,
    TaskStatus,
)
from specforge.modeling.target.remote_backend.mooncake_client import (
    EagleMooncakeStore,
    MooncakeConfig,
)
from specforge.modeling.target.remote_backend.remote_target_model import (
    RemoteBackendConfig,
    RemoteEagle3TargetModel,
)
from specforge.modeling.target.remote_backend.task_queue import (
    NotificationPublisher,
    NotificationSubscriber,
    QueueConfig,
    TaskConsumer,
    TaskProducer,
)


@pytest.fixture
def integration_config():
    """Create configuration for integration tests."""
    import random

    base_port = 16000 + random.randint(0, 1000)
    
    queue_config = QueueConfig(
        task_queue_addr=f"tcp://127.0.0.1:{base_port}",
        notify_addr=f"tcp://127.0.0.1:{base_port + 1}",
    )
    
    mooncake_config = MooncakeConfig()
    
    return queue_config, mooncake_config


class MockInferenceWorker:
    """
    Mock inference worker for testing.
    
    This simulates the inference worker by:
    1. Pulling tasks from the queue
    2. Generating fake hidden states
    3. Storing results in Mooncake Store
    4. Publishing notifications
    """

    def __init__(self, queue_config: QueueConfig, mooncake_config: MooncakeConfig):
        self.queue_config = queue_config
        self.mooncake_config = mooncake_config
        self._running = False
        self._thread = None

    def start(self):
        """Start the mock worker in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the mock worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        """Worker main loop."""
        consumer = TaskConsumer(self.queue_config)
        consumer.bind()

        publisher = NotificationPublisher(self.queue_config)
        publisher.bind()

        store = EagleMooncakeStore(self.mooncake_config)
        store.setup()

        time.sleep(0.1)

        while self._running:
            task = consumer.pull(timeout_ms=100)
            if task is None:
                continue

            try:
                input_ids = task.get_input_ids()
                attention_mask = task.get_attention_mask()
                loss_mask = task.get_loss_mask()

                batch_size = input_ids.shape[0] if input_ids.dim() > 1 else 1
                seq_len = input_ids.shape[-1]
                hidden_dim = 128
                vocab_size = 100

                hidden_states = torch.randn(
                    batch_size, seq_len, hidden_dim * 3, dtype=torch.bfloat16
                )
                target = torch.randn(
                    batch_size, seq_len, vocab_size, dtype=torch.bfloat16
                )

                store.put_eagle3_output(
                    key=task.task_id,
                    hidden_states=hidden_states,
                    target=target,
                    loss_mask=loss_mask.unsqueeze(-1) if loss_mask.dim() == 2 else loss_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                publisher.publish(
                    TaskNotification(
                        task_id=task.task_id,
                        status=TaskStatus.READY,
                        mooncake_key=task.task_id,
                    )
                )

            except Exception as e:
                publisher.publish(
                    TaskNotification(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error_message=str(e),
                    )
                )

        consumer.close()
        publisher.close()
        store.close()


class TestEndToEndFlow:
    """End-to-end tests for the distributed inference pipeline."""

    def test_single_task_flow(self, integration_config):
        """Test complete flow for a single task."""
        queue_config, mooncake_config = integration_config

        worker = MockInferenceWorker(queue_config, mooncake_config)
        worker.start()

        time.sleep(0.2)

        producer = TaskProducer(queue_config)
        producer.connect()

        subscriber = NotificationSubscriber(queue_config)
        subscriber.connect()
        subscriber.start_listener()

        store = EagleMooncakeStore(mooncake_config)
        store.setup()

        try:
            input_ids = torch.randint(0, 100, (1, 10))
            attention_mask = torch.ones(1, 10)
            loss_mask = torch.ones(1, 10)

            task = InferenceTask.create(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                task_id="e2e-task-1",
            )

            subscriber.subscribe(task.task_id)

            producer.push(task)

            notification = subscriber.wait_for(task.task_id, timeout=5.0)

            assert notification is not None
            assert notification.status == TaskStatus.READY
            assert notification.mooncake_key == "e2e-task-1"

            output = store.get_eagle3_output(notification.mooncake_key, device="cpu")

            assert output.hidden_states is not None
            assert output.target is not None
            assert output.hidden_states.shape[0] == 1
            assert output.hidden_states.shape[1] == 10

        finally:
            worker.stop()
            producer.close()
            subscriber.close()
            store.close()

    def test_multiple_tasks_flow(self, integration_config):
        """Test complete flow for multiple tasks."""
        queue_config, mooncake_config = integration_config

        worker = MockInferenceWorker(queue_config, mooncake_config)
        worker.start()

        time.sleep(0.2)

        producer = TaskProducer(queue_config)
        producer.connect()

        subscriber = NotificationSubscriber(queue_config)
        subscriber.connect()
        subscriber.start_listener()

        store = EagleMooncakeStore(mooncake_config)
        store.setup()

        try:
            num_tasks = 3
            task_ids = []

            for i in range(num_tasks):
                input_ids = torch.randint(0, 100, (1, 10 + i))
                attention_mask = torch.ones(1, 10 + i)
                loss_mask = torch.ones(1, 10 + i)

                task = InferenceTask.create(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                    task_id=f"batch-task-{i}",
                )
                task_ids.append(task.task_id)
                subscriber.subscribe(task.task_id)
                producer.push(task)

            completed = 0
            for task_id in task_ids:
                notification = subscriber.wait_for(task_id, timeout=5.0)
                if notification and notification.status == TaskStatus.READY:
                    output = store.get_eagle3_output(
                        notification.mooncake_key, device="cpu"
                    )
                    assert output is not None
                    completed += 1

            assert completed == num_tasks

        finally:
            worker.stop()
            producer.close()
            subscriber.close()
            store.close()


class TestRemoteTargetModelIntegration:
    """Integration tests for RemoteEagle3TargetModel."""

    def test_remote_model_with_mock_worker(self, integration_config):
        """Test RemoteEagle3TargetModel with mock worker."""
        queue_config, mooncake_config = integration_config

        worker = MockInferenceWorker(queue_config, mooncake_config)
        worker.start()

        time.sleep(0.2)

        config = RemoteBackendConfig(
            task_queue_addr=queue_config.task_queue_addr,
            notify_addr=queue_config.notify_addr,
            task_timeout=5.0,
            mooncake_config=mooncake_config,
            queue_config=queue_config,
        )

        model = RemoteEagle3TargetModel(config)

        try:
            model.connect()

            input_ids = torch.randint(0, 100, (1, 10))
            attention_mask = torch.ones(1, 10)
            loss_mask = torch.ones(1, 10)

            output = model.generate_eagle3_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )

            assert output is not None
            assert output.hidden_states is not None
            assert output.target is not None
            assert output.input_ids is not None

        finally:
            model.disconnect()
            worker.stop()

    def test_remote_model_context_manager(self, integration_config):
        """Test RemoteEagle3TargetModel as context manager."""
        queue_config, mooncake_config = integration_config

        worker = MockInferenceWorker(queue_config, mooncake_config)
        worker.start()

        time.sleep(0.2)

        config = RemoteBackendConfig(
            task_queue_addr=queue_config.task_queue_addr,
            notify_addr=queue_config.notify_addr,
            task_timeout=5.0,
            mooncake_config=mooncake_config,
            queue_config=queue_config,
        )

        try:
            with RemoteEagle3TargetModel(config) as model:
                input_ids = torch.randint(0, 100, (1, 8))
                attention_mask = torch.ones(1, 8)
                loss_mask = torch.ones(1, 8)

                output = model.generate_eagle3_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                )

                assert output is not None

        finally:
            worker.stop()
