"""Tests for ZeroMQ task queue components."""

import threading
import time

import pytest
import torch

from specforge.modeling.target.remote_backend.messages import (
    InferenceTask,
    TaskNotification,
    TaskStatus,
)
from specforge.modeling.target.remote_backend.task_queue import (
    NotificationPublisher,
    NotificationSubscriber,
    QueueConfig,
    TaskConsumer,
    TaskProducer,
)


@pytest.fixture
def queue_config():
    """Create a queue config with unique ports for test isolation."""
    import random

    base_port = 15000 + random.randint(0, 1000)
    return QueueConfig(
        task_queue_addr=f"tcp://127.0.0.1:{base_port}",
        notify_addr=f"tcp://127.0.0.1:{base_port + 1}",
    )


class TestTaskQueueBasic:
    """Basic tests for task queue without actual networking."""

    def test_queue_config_defaults(self):
        config = QueueConfig()
        assert config.task_queue_addr == "tcp://localhost:5555"
        assert config.notify_addr == "tcp://localhost:5556"
        assert config.task_timeout_ms == 300000

    def test_producer_creation(self, queue_config):
        producer = TaskProducer(queue_config)
        assert producer._connected is False
        producer.close()

    def test_consumer_creation(self, queue_config):
        consumer = TaskConsumer(queue_config)
        assert consumer._bound is False
        consumer.close()


class TestTaskQueueIntegration:
    """Integration tests for task queue with actual ZeroMQ communication."""

    def test_push_pull_task(self, queue_config):
        """Test basic task push/pull through the queue."""
        consumer = TaskConsumer(queue_config)
        consumer.bind()

        producer = TaskProducer(queue_config)
        producer.connect()

        time.sleep(0.1)

        input_ids = torch.randint(0, 1000, (1, 10))
        task = InferenceTask.create(
            input_ids=input_ids,
            attention_mask=torch.ones(1, 10),
            loss_mask=torch.ones(1, 10),
            task_id="test-task-1",
        )

        producer.push(task)

        received_task = consumer.pull(timeout_ms=1000)

        assert received_task is not None
        assert received_task.task_id == "test-task-1"
        assert torch.equal(received_task.get_input_ids(), input_ids)

        producer.close()
        consumer.close()

    def test_pull_timeout(self, queue_config):
        """Test that pull returns None on timeout."""
        consumer = TaskConsumer(queue_config)
        consumer.bind()

        result = consumer.pull(timeout_ms=100)
        assert result is None

        consumer.close()

    def test_multiple_tasks(self, queue_config):
        """Test pushing and pulling multiple tasks."""
        consumer = TaskConsumer(queue_config)
        consumer.bind()

        producer = TaskProducer(queue_config)
        producer.connect()

        time.sleep(0.1)

        num_tasks = 5
        for i in range(num_tasks):
            task = InferenceTask.create(
                input_ids=torch.randint(0, 1000, (1, 10)),
                attention_mask=torch.ones(1, 10),
                loss_mask=torch.ones(1, 10),
                task_id=f"task-{i}",
            )
            producer.push(task)

        received_ids = []
        for _ in range(num_tasks):
            task = consumer.pull(timeout_ms=1000)
            assert task is not None
            received_ids.append(task.task_id)

        assert set(received_ids) == {f"task-{i}" for i in range(num_tasks)}

        producer.close()
        consumer.close()


class TestNotificationPubSub:
    """Tests for notification publish/subscribe."""

    def test_publish_subscribe_notification(self, queue_config):
        """Test basic notification pub/sub."""
        publisher = NotificationPublisher(queue_config)
        publisher.bind()

        subscriber = NotificationSubscriber(queue_config)
        subscriber.connect()
        subscriber.subscribe("task-123")

        time.sleep(0.2)

        notification = TaskNotification(
            task_id="task-123",
            status=TaskStatus.READY,
            mooncake_key="task-123",
        )
        publisher.publish(notification)

        received = subscriber.receive(timeout_ms=1000)

        assert received is not None
        assert received.task_id == "task-123"
        assert received.status == TaskStatus.READY

        publisher.close()
        subscriber.close()

    def test_subscribe_specific_task(self, queue_config):
        """Test subscribing to specific task IDs."""
        publisher = NotificationPublisher(queue_config)
        publisher.bind()

        subscriber = NotificationSubscriber(queue_config)
        subscriber.connect()
        subscriber.subscribe("task-A")

        time.sleep(0.2)

        publisher.publish(
            TaskNotification(task_id="task-B", status=TaskStatus.READY)
        )
        publisher.publish(
            TaskNotification(task_id="task-A", status=TaskStatus.READY)
        )

        received = subscriber.receive(timeout_ms=1000)
        assert received is not None
        assert received.task_id == "task-A"

        publisher.close()
        subscriber.close()

    def test_listener_thread(self, queue_config):
        """Test background listener thread."""
        publisher = NotificationPublisher(queue_config)
        publisher.bind()

        subscriber = NotificationSubscriber(queue_config)
        subscriber.connect()
        subscriber.start_listener()

        time.sleep(0.2)

        subscriber.subscribe("async-task")

        notification = TaskNotification(
            task_id="async-task",
            status=TaskStatus.READY,
            mooncake_key="async-task",
        )
        publisher.publish(notification)

        received = subscriber.wait_for("async-task", timeout=2.0)

        assert received is not None
        assert received.task_id == "async-task"
        assert received.status == TaskStatus.READY

        subscriber.stop_listener()
        publisher.close()
        subscriber.close()

    def test_wait_for_timeout(self, queue_config):
        """Test wait_for returns None on timeout."""
        subscriber = NotificationSubscriber(queue_config)
        subscriber.connect()
        subscriber.start_listener()

        subscriber.subscribe("nonexistent-task")

        result = subscriber.wait_for("nonexistent-task", timeout=0.1)
        assert result is None

        subscriber.stop_listener()
        subscriber.close()


class TestConcurrentAccess:
    """Tests for concurrent task queue access."""

    def test_multiple_producers(self, queue_config):
        """Test multiple producers sending to one consumer."""
        consumer = TaskConsumer(queue_config)
        consumer.bind()

        producers = []
        for i in range(3):
            p = TaskProducer(queue_config)
            p.connect()
            producers.append(p)

        time.sleep(0.1)

        for i, producer in enumerate(producers):
            task = InferenceTask.create(
                input_ids=torch.randint(0, 1000, (1, 10)),
                attention_mask=torch.ones(1, 10),
                loss_mask=torch.ones(1, 10),
                task_id=f"producer-{i}-task",
            )
            producer.push(task)

        received_ids = []
        for _ in range(3):
            task = consumer.pull(timeout_ms=1000)
            if task:
                received_ids.append(task.task_id)

        assert len(received_ids) == 3

        for p in producers:
            p.close()
        consumer.close()
