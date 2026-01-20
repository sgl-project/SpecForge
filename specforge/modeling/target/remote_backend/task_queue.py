"""
ZeroMQ-based task queue and notification system for distributed inference.

Components:
- TaskProducer: PUSH socket for submitting tasks (training side)
- TaskConsumer: PULL socket for receiving tasks (worker side)
- NotificationPublisher: PUB socket for broadcasting completions (worker side)
- NotificationSubscriber: SUB socket with topic filtering (training side)
- TaskQueueBroker: Optional PULL/PUSH broker for task distribution
- NotificationBroker: Optional XSUB/XPUB broker for notifications
"""

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import zmq

from .messages import InferenceTask, TaskNotification

logger = logging.getLogger(__name__)


@dataclass
class QueueConfig:
    """Configuration for the task queue and notification system."""

    task_queue_addr: str = "tcp://localhost:5555"
    notify_addr: str = "tcp://localhost:5556"
    task_timeout_ms: int = 300000
    hwm: int = 10000


class TaskProducer:
    """
    PUSH socket for submitting inference tasks.

    Training ranks use this to send tasks to inference workers.
    When multiple workers are available, ZMQ load-balances across them.
    """

    def __init__(self, config: QueueConfig):
        self.config = config
        self._ctx = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._connected = False

    def connect(self, addr: Optional[str] = None) -> None:
        """Connect to the task queue (worker's PULL socket or broker)."""
        if self._connected:
            return

        addr = addr or self.config.task_queue_addr
        self._socket = self._ctx.socket(zmq.PUSH)
        self._socket.set_hwm(self.config.hwm)
        self._socket.connect(addr)
        self._connected = True
        logger.debug(f"TaskProducer connected to {addr}")

    def push(self, task: InferenceTask) -> None:
        """Push a task to the queue."""
        if not self._connected or self._socket is None:
            raise RuntimeError("TaskProducer not connected")
        self._socket.send(task.serialize())

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            self._socket.close(linger=100)
            self._socket = None
        self._connected = False


class TaskConsumer:
    """
    PULL socket for receiving inference tasks.

    Inference workers use this to pull tasks from the queue.
    Multiple workers can pull from the same address for load balancing.
    """

    def __init__(self, config: QueueConfig):
        self.config = config
        self._ctx = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._bound = False
        self._connected = False

    def bind(self, addr: Optional[str] = None) -> None:
        """Bind to address (when workers pull directly from training)."""
        if self._bound or self._connected:
            return

        addr = addr or self.config.task_queue_addr
        self._socket = self._ctx.socket(zmq.PULL)
        self._socket.set_hwm(self.config.hwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(addr)
        self._bound = True
        logger.debug(f"TaskConsumer bound to {addr}")

    def connect(self, addr: Optional[str] = None) -> None:
        """Connect to address (when using broker or specific endpoint)."""
        if self._bound or self._connected:
            return

        addr = addr or self.config.task_queue_addr
        self._socket = self._ctx.socket(zmq.PULL)
        self._socket.set_hwm(self.config.hwm)
        self._socket.connect(addr)
        self._connected = True
        logger.debug(f"TaskConsumer connected to {addr}")

    def pull(self, timeout_ms: Optional[int] = None) -> Optional[InferenceTask]:
        """Pull a task from the queue. Returns None on timeout."""
        if self._socket is None:
            raise RuntimeError("TaskConsumer not bound or connected")

        timeout_ms = (
            timeout_ms if timeout_ms is not None else self.config.task_timeout_ms
        )

        if self._socket.poll(timeout_ms):
            data = self._socket.recv()
            return InferenceTask.deserialize(data)
        return None

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            self._socket.close(linger=100)
            self._socket = None
        self._bound = False
        self._connected = False


class NotificationPublisher:
    """
    PUB socket for broadcasting task completion notifications.

    Inference workers publish notifications with task_id as the topic.
    Training ranks subscribe to specific task_ids they care about.
    """

    def __init__(self, config: QueueConfig):
        self.config = config
        self._ctx = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._bound = False
        self._connected = False

    def bind(self, addr: Optional[str] = None) -> None:
        """Bind to address (workers bind, training connects)."""
        if self._bound or self._connected:
            return

        addr = addr or self.config.notify_addr
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.set_hwm(self.config.hwm)
        self._socket.bind(addr)
        self._bound = True
        logger.debug(f"NotificationPublisher bound to {addr}")

    def connect(self, addr: Optional[str] = None) -> None:
        """Connect to broker (when using notification broker)."""
        if self._bound or self._connected:
            return

        addr = addr or self.config.notify_addr
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.set_hwm(self.config.hwm)
        self._socket.connect(addr)
        self._connected = True
        logger.debug(f"NotificationPublisher connected to {addr}")

    def publish(self, notification: TaskNotification) -> None:
        """Publish a notification with task_id as topic."""
        if self._socket is None:
            raise RuntimeError("NotificationPublisher not bound or connected")

        topic = notification.task_id.encode("utf-8")
        payload = notification.serialize()
        self._socket.send_multipart([topic, payload])

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            self._socket.close(linger=100)
            self._socket = None
        self._bound = False
        self._connected = False


class NotificationSubscriber:
    """
    SUB socket for receiving task completion notifications.

    Training ranks subscribe to specific task_ids and wait for notifications.
    Supports both synchronous receive() and async wait_for() with background listener.
    """

    def __init__(self, config: QueueConfig):
        self.config = config
        self._ctx = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._connected = False

        self._listener_thread: Optional[threading.Thread] = None
        self._listener_running = False
        self._notifications: Dict[str, TaskNotification] = {}
        self._notification_events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def connect(self, addr: Optional[str] = None) -> None:
        """Connect to notification publisher."""
        if self._connected:
            return

        addr = addr or self.config.notify_addr
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.set_hwm(self.config.hwm)
        self._socket.connect(addr)
        self._connected = True
        logger.debug(f"NotificationSubscriber connected to {addr}")

    def subscribe(self, task_id: str) -> None:
        """Subscribe to notifications for a specific task_id."""
        if self._socket is None:
            raise RuntimeError("NotificationSubscriber not connected")

        self._socket.subscribe(task_id.encode("utf-8"))

        with self._lock:
            if task_id not in self._notification_events:
                self._notification_events[task_id] = threading.Event()

    def unsubscribe(self, task_id: str) -> None:
        """Unsubscribe from a task_id."""
        if self._socket is None:
            return

        self._socket.unsubscribe(task_id.encode("utf-8"))

        with self._lock:
            self._notifications.pop(task_id, None)
            self._notification_events.pop(task_id, None)

    def receive(self, timeout_ms: int = 1000) -> Optional[TaskNotification]:
        """Receive a notification (blocking with timeout)."""
        if self._socket is None:
            raise RuntimeError("NotificationSubscriber not connected")

        if self._socket.poll(timeout_ms):
            topic, payload = self._socket.recv_multipart()
            return TaskNotification.deserialize(payload)
        return None

    def start_listener(self) -> None:
        """Start background thread to listen for notifications."""
        if self._listener_running:
            return

        self._listener_running = True
        self._listener_thread = threading.Thread(
            target=self._listener_loop, daemon=True, name="notification-listener"
        )
        self._listener_thread.start()
        logger.debug("NotificationSubscriber listener started")

    def stop_listener(self) -> None:
        """Stop the background listener thread."""
        self._listener_running = False
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=1.0)
            self._listener_thread = None

    def _listener_loop(self) -> None:
        """Background loop to receive notifications and store them."""
        while self._listener_running:
            if self._socket is None:
                break

            try:
                if self._socket.poll(100):
                    topic, payload = self._socket.recv_multipart()
                    task_id = topic.decode("utf-8")
                    notification = TaskNotification.deserialize(payload)

                    with self._lock:
                        self._notifications[task_id] = notification
                        if task_id in self._notification_events:
                            self._notification_events[task_id].set()

            except zmq.ZMQError as e:
                if self._listener_running:
                    logger.warning(f"ZMQ error in listener: {e}")
                break

    def wait_for(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Optional[TaskNotification]:
        """Wait for a notification for a specific task_id."""
        with self._lock:
            if task_id in self._notifications:
                return self._notifications.pop(task_id)

            if task_id not in self._notification_events:
                self._notification_events[task_id] = threading.Event()
            event = self._notification_events[task_id]

        if event.wait(timeout=timeout):
            with self._lock:
                return self._notifications.pop(task_id, None)
        return None

    def poll(self, task_id: str, timeout: float = 0.0) -> Optional[TaskNotification]:
        """Non-blocking check if notification is available."""
        with self._lock:
            if task_id in self._notifications:
                return self._notifications.pop(task_id)

        if timeout > 0:
            with self._lock:
                if task_id not in self._notification_events:
                    self._notification_events[task_id] = threading.Event()
                event = self._notification_events[task_id]

            if event.wait(timeout=timeout):
                with self._lock:
                    return self._notifications.pop(task_id, None)

        return None

    def close(self) -> None:
        """Close the subscriber and stop listener."""
        self.stop_listener()
        if self._socket is not None:
            self._socket.close(linger=100)
            self._socket = None
        self._connected = False

        with self._lock:
            self._notifications.clear()
            self._notification_events.clear()


class TaskQueueBroker:
    """
    Optional broker for task distribution (PULL/PUSH pattern).

    Use when you want to decouple training nodes from worker addresses.
    Training nodes PUSH to broker, workers PULL from broker.

    Training ──PUSH──► Broker ──PUSH──► Workers (PULL)
    """

    def __init__(
        self,
        frontend_addr: str = "tcp://*:5555",
        backend_addr: str = "tcp://*:5554",
    ):
        self.frontend_addr = frontend_addr
        self.backend_addr = backend_addr
        self._ctx = zmq.Context.instance()
        self._frontend: Optional[zmq.Socket] = None
        self._backend: Optional[zmq.Socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the broker in a background thread."""
        if self._running:
            return

        self._frontend = self._ctx.socket(zmq.PULL)
        self._frontend.bind(self.frontend_addr)

        self._backend = self._ctx.socket(zmq.PUSH)
        self._backend.bind(self.backend_addr)

        self._running = True
        self._thread = threading.Thread(
            target=self._broker_loop, daemon=True, name="task-queue-broker"
        )
        self._thread.start()
        logger.info(
            f"TaskQueueBroker started: frontend={self.frontend_addr}, backend={self.backend_addr}"
        )

    def _broker_loop(self) -> None:
        """Forward tasks from frontend to backend."""
        poller = zmq.Poller()
        poller.register(self._frontend, zmq.POLLIN)

        while self._running:
            try:
                socks = dict(poller.poll(100))
                if self._frontend in socks:
                    msg = self._frontend.recv()
                    self._backend.send(msg)
            except zmq.ZMQError:
                if self._running:
                    raise
                break

    def stop(self) -> None:
        """Stop the broker."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._frontend is not None:
            self._frontend.close(linger=0)
            self._frontend = None
        if self._backend is not None:
            self._backend.close(linger=0)
            self._backend = None


class NotificationBroker:
    """
    Optional broker for notifications (XSUB/XPUB pattern).

    Use when workers and training nodes can't directly connect.
    Workers PUB to broker, training nodes SUB from broker.

    Workers (PUB) ──► Broker (XSUB/XPUB) ──► Training (SUB)
    """

    def __init__(
        self,
        frontend_addr: str = "tcp://*:5556",
        backend_addr: str = "tcp://*:5557",
    ):
        self.frontend_addr = frontend_addr
        self.backend_addr = backend_addr
        self._ctx = zmq.Context.instance()
        self._frontend: Optional[zmq.Socket] = None
        self._backend: Optional[zmq.Socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the broker in a background thread."""
        if self._running:
            return

        self._frontend = self._ctx.socket(zmq.XSUB)
        self._frontend.bind(self.frontend_addr)

        self._backend = self._ctx.socket(zmq.XPUB)
        self._backend.bind(self.backend_addr)

        self._running = True
        self._thread = threading.Thread(
            target=self._broker_loop, daemon=True, name="notification-broker"
        )
        self._thread.start()
        logger.info(
            f"NotificationBroker started: frontend={self.frontend_addr}, backend={self.backend_addr}"
        )

    def _broker_loop(self) -> None:
        """Proxy messages between XSUB and XPUB."""
        poller = zmq.Poller()
        poller.register(self._frontend, zmq.POLLIN)
        poller.register(self._backend, zmq.POLLIN)

        while self._running:
            try:
                socks = dict(poller.poll(100))

                if self._frontend in socks:
                    msg = self._frontend.recv_multipart()
                    self._backend.send_multipart(msg)

                if self._backend in socks:
                    msg = self._backend.recv_multipart()
                    self._frontend.send_multipart(msg)

            except zmq.ZMQError:
                if self._running:
                    raise
                break

    def stop(self) -> None:
        """Stop the broker."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._frontend is not None:
            self._frontend.close(linger=0)
            self._frontend = None
        if self._backend is not None:
            self._backend.close(linger=0)
            self._backend = None
