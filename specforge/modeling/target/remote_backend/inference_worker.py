"""
Inference worker service for distributed Eagle3 training.

This module provides the InferenceWorker class that runs on inference nodes,
processing tasks from the queue using SGLangRunner and storing
results in Mooncake Store.

For tensor parallelism (TP > 1), launch with torchrun:
    torchrun --standalone --nproc_per_node=4 -m specforge.modeling.target.remote_backend ...

For single GPU (TP = 1), launch directly:
    python -m specforge.modeling.target.remote_backend ...
"""

import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist

from ..eagle3_target_model import SGLangEagle3TargetModel
from ..sglang_backend import SGLangBackendArgs
from .messages import (
    Eagle3OutputData,
    InferenceTask,
    TaskNotification,
    TaskStatus,
)
from .mooncake_client import EagleMooncakeStore, MooncakeConfig, Eagle3HostBufferWriter
from .task_queue import (
    NotificationPublisher,
    QueueConfig,
    TaskConsumer,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceWorkerConfig:
    """Configuration for the inference worker."""

    model_path: str
    task_queue_addr: str = "tcp://localhost:5555"
    notify_addr: str = "tcp://localhost:5556"

    tp_size: int = 1
    dtype: str = "bfloat16"
    trust_remote_code: bool = False

    num_workers: int = 1
    worker_id: int = 0

    use_zero_copy: bool = True

    mooncake_config: MooncakeConfig = None
    queue_config: QueueConfig = None
    sglang_backend_args: SGLangBackendArgs = None

    aux_hidden_states_layers: Optional[List[int]] = None

    def __post_init__(self):
        if self.mooncake_config is None:
            self.mooncake_config = MooncakeConfig()
        if self.queue_config is None:
            self.queue_config = QueueConfig(
                task_queue_addr=self.task_queue_addr,
                notify_addr=self.notify_addr,
            )
        if self.sglang_backend_args is None:
            self.sglang_backend_args = SGLangBackendArgs(tp_size=self.tp_size)
        else:
            self.sglang_backend_args.tp_size = self.tp_size


class InferenceWorker:
    """
    Worker service that processes inference tasks using SGLangRunner.
    
    This worker:
    1. Pulls tasks from the ZeroMQ task queue
    2. Runs inference using SGLangEagle3TargetModel (backed by SGLangRunner)
    3. Extracts hidden states from the model
    4. Stores results in Mooncake Store
    5. Publishes completion notifications
    """

    def __init__(self, config: InferenceWorkerConfig):
        self.config = config
        self.target_model: Optional[SGLangEagle3TargetModel] = None
        self.task_consumer: Optional[TaskConsumer] = None
        self.notification_pub: Optional[NotificationPublisher] = None
        self.mooncake_store: Optional[EagleMooncakeStore] = None
        self.host_buffer_writer: Optional[Eagle3HostBufferWriter] = None

        self._running = False
        self._shutdown_event = threading.Event()

    def setup(self) -> None:
        """Initialize all components."""
        tp_rank = self._get_tp_rank()
        print(f"[SETUP] Setting up InferenceWorker {self.config.worker_id} (tp_rank={tp_rank}/{self.config.tp_size})", flush=True)

        self._setup_target_model()
        
        if self._is_main_rank():
            print(f"[SETUP] Setting up task queue...", flush=True)
            self._setup_task_queue()
            print(f"[SETUP] Setting up notification...", flush=True)
            self._setup_notification()
            print(f"[SETUP] Setting up mooncake...", flush=True)
            self._setup_mooncake()

            if self.config.use_zero_copy:
                self.host_buffer_writer = Eagle3HostBufferWriter(
                    max_buffer_size=4 * 1024**3,
                )
                print("[SETUP] Host buffer writer initialized", flush=True)

        print("[SETUP] InferenceWorker setup complete", flush=True)

    def _setup_target_model(self) -> None:
        """Initialize SGLangEagle3TargetModel using SGLangRunner."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        sglang_kwargs = self.config.sglang_backend_args.to_kwargs()

        self.target_model = SGLangEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=self.config.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            **sglang_kwargs,
        )
        self.target_model.set_aux_hidden_states_layers(
            self.config.aux_hidden_states_layers
        )

        logger.info(f"SGLangEagle3TargetModel initialized with model: {self.config.model_path}")
    
    def _get_tp_rank(self) -> int:
        """Get tensor parallel rank from distributed context or environment."""
        if dist.is_initialized():
            return dist.get_rank()
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank)
        return 0
    
    def _is_main_rank(self) -> bool:
        """Check if this is the main rank (rank 0) for communication."""
        return self._get_tp_rank() == 0

    def _setup_task_queue(self) -> None:
        """Initialize task queue consumer."""
        self.task_consumer = TaskConsumer(self.config.queue_config)
        self.task_consumer.bind(self.config.task_queue_addr)
        print(f"[SETUP] Task queue bound to {self.config.task_queue_addr}", flush=True)

    def _setup_notification(self) -> None:
        """Initialize notification publisher."""
        self.notification_pub = NotificationPublisher(self.config.queue_config)
        self.notification_pub.bind(self.config.notify_addr)
        print(f"[SETUP] Notification publisher bound to {self.config.notify_addr}", flush=True)

    def _setup_mooncake(self) -> None:
        """Initialize Mooncake Store client."""
        self.mooncake_store = EagleMooncakeStore(self.config.mooncake_config)
        self.mooncake_store.setup()
        logger.info("Mooncake Store client initialized")

    def run(self) -> None:
        """Main worker loop."""
        self._running = True
        tp_rank = self._get_tp_rank()
        print(f"[RUN] InferenceWorker {self.config.worker_id} starting main loop (tp_rank={tp_rank})", flush=True)
        print(f"[RUN] Waiting for tasks on {self.config.task_queue_addr}...", flush=True)

        while self._running and not self._shutdown_event.is_set():
            try:
                task = self._get_next_task()
                if task is None:
                    continue

                print(f"[RUN] Received task: {task.task_id}", flush=True)
                self._process_task(task)

            except Exception as e:
                print(f"[RUN] Error in worker loop: {e}", flush=True)
                logger.error(f"Error in worker loop: {e}", exc_info=True)

        print(f"[RUN] InferenceWorker {self.config.worker_id} stopped", flush=True)
    
    def _get_next_task(self) -> Optional[InferenceTask]:
        """
        Get the next task to process.
        
        For TP > 1, rank 0 pulls the task and broadcasts to other ranks.
        """
        if self.config.tp_size <= 1:
            return self.task_consumer.pull(timeout_ms=1000)
        
        task_data = [None]
        
        if self._is_main_rank():
            task = self.task_consumer.pull(timeout_ms=1000)
            if task is not None:
                task_data[0] = task.serialize()
        
        dist.broadcast_object_list(task_data, src=0)
        
        if task_data[0] is None:
            return None
        
        return InferenceTask.deserialize(task_data[0])

    def _process_task(self, task: InferenceTask) -> None:
        """Process a single inference task. All TP ranks participate in inference."""
        logger.debug(f"Processing task {task.task_id}")
        start_time = time.time()

        try:
            print(f"[TASK] Deserializing tensors...", flush=True)
            input_ids = task.get_input_ids()
            attention_mask = task.get_attention_mask()
            loss_mask = task.get_loss_mask()
            print(f"[TASK] input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}", flush=True)

            if task.aux_hidden_states_layers is not None:
                self.target_model.set_aux_hidden_states_layers(
                    task.aux_hidden_states_layers
                )

            print(f"[TASK] Running inference...", flush=True)
            output = self._run_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                return_last_hidden_states=task.return_last_hidden_states,
                return_logits=task.return_logits,
            )
            print(f"[TASK] Inference complete", flush=True)

            if not self._is_main_rank():
                return

            hidden_states = output.hidden_states
            target = output.target
            last_hidden_states = output.last_hidden_states

            if self.config.use_zero_copy and self.host_buffer_writer is not None:
                host_buffer, data_size, shapes = self.host_buffer_writer.pack_eagle3_output(
                    hidden_states=hidden_states,
                    target=target,
                    loss_mask=loss_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    last_hidden_states=last_hidden_states,
                )
                self.mooncake_store.put_from_host_buffer(task.task_id, host_buffer, data_size)
            else:
                output_data = Eagle3OutputData.from_tensors(
                    hidden_states=hidden_states,
                    target=target,
                    loss_mask=loss_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    last_hidden_states=last_hidden_states,
                )
                self.mooncake_store.put_eagle3_output_data(task.task_id, output_data)
                shapes = {
                    "hidden_states": tuple(hidden_states.shape),
                    "target": tuple(target.shape) if target is not None else None,
                    "loss_mask": tuple(loss_mask.shape),
                    "input_ids": tuple(input_ids.shape),
                    "attention_mask": tuple(attention_mask.shape),
                }
                if last_hidden_states is not None:
                    shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
                data_size = 0

            self.notification_pub.publish(
                TaskNotification(
                    task_id=task.task_id,
                    status=TaskStatus.READY,
                    mooncake_key=task.task_id,
                    tensor_shapes=shapes,
                    data_size=data_size,
                )
            )

            elapsed = time.time() - start_time
            logger.info(f"Task {task.task_id} completed in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)

            if self._is_main_rank():
                self.notification_pub.publish(
                    TaskNotification(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error_message=str(e),
                    )
                )

    def _run_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
    ):
        """
        Run inference using SGLangEagle3TargetModel.
        
        Returns:
            Eagle3TargetOutput containing hidden_states, target, loss_mask, etc.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if loss_mask.dim() == 1:
            loss_mask = loss_mask.unsqueeze(0)

        print(f"[INFERENCE] Calling target_model.extend()...", flush=True)
        data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list = (
            self.target_model.extend(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                return_last_hidden_states=return_last_hidden_states,
                return_logits=return_logits,
            )
        )
        print(f"[INFERENCE] target_model.extend() returned", flush=True)

        from ..eagle3_target_model import Eagle3TargetOutput
        from specforge.utils import padding

        aux_hidden_states_out = []
        target_out = []
        loss_mask_out = []
        input_ids_out = []
        last_hidden_states_out = []

        for data, logits, aux_hidden_states, last_hidden_states in zip(
            data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list
        ):
            aux_hidden_states_out.append(aux_hidden_states.unsqueeze(0))
            loss_mask_out.append(data[2])
            input_ids_out.append(data[0])

            if logits is not None:
                target_out.append(logits.unsqueeze(0))
            else:
                target_out.append(None)

            if last_hidden_states is not None:
                last_hidden_states_out.append(last_hidden_states.unsqueeze(0))
            else:
                last_hidden_states_out.append(None)

        aux_hidden_states_out = torch.cat(aux_hidden_states_out, dim=0)
        loss_mask_out = torch.cat(loss_mask_out, dim=0)
        input_ids_out = torch.cat(input_ids_out, dim=0)

        if target_out[0] is not None:
            target_out = torch.cat(target_out, dim=0)
        else:
            target_out = None

        if last_hidden_states_out[0] is not None:
            last_hidden_states_out = torch.cat(last_hidden_states_out, dim=0)
        else:
            last_hidden_states_out = None

        target_out = padding(target_out, left=False)
        input_ids_out = padding(input_ids_out, left=False)
        loss_mask_out = loss_mask_out[..., None]

        return Eagle3TargetOutput(
            hidden_states=aux_hidden_states_out,
            target=target_out,
            loss_mask=loss_mask_out,
            input_ids=input_ids_out,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states_out,
        )

    def shutdown(self) -> None:
        """Gracefully shutdown the worker."""
        logger.info(f"Shutting down InferenceWorker {self.config.worker_id}")
        self._running = False
        self._shutdown_event.set()

        if self.target_model is not None:
            self.target_model = None

        if self.task_consumer is not None:
            self.task_consumer.close()
            self.task_consumer = None

        if self.notification_pub is not None:
            self.notification_pub.close()
            self.notification_pub = None

        if self.mooncake_store is not None:
            self.mooncake_store.close()
            self.mooncake_store = None

        logger.info(f"InferenceWorker {self.config.worker_id} shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


class InferenceWorkerManager:
    """
    Manager for running multiple inference workers.
    
    Useful for running multiple workers in a single process
    (e.g., for testing or single-node multi-GPU setups).
    """

    def __init__(self, configs: List[InferenceWorkerConfig]):
        self.configs = configs
        self.workers: List[InferenceWorker] = []
        self.threads: List[threading.Thread] = []
        self._running = False

    def start(self) -> None:
        """Start all workers in separate threads."""
        self._running = True

        for config in self.configs:
            worker = InferenceWorker(config)
            worker.setup()
            self.workers.append(worker)

            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            self.threads.append(thread)

        logger.info(f"Started {len(self.workers)} inference workers")

    def stop(self) -> None:
        """Stop all workers."""
        self._running = False

        for worker in self.workers:
            worker.shutdown()

        for thread in self.threads:
            thread.join(timeout=5.0)

        self.workers.clear()
        self.threads.clear()

        logger.info("All inference workers stopped")

    def wait(self) -> None:
        """Wait for all workers to complete."""
        for thread in self.threads:
            thread.join()


def _init_distributed_for_tp(config: InferenceWorkerConfig) -> int:
    """
    Initialize distributed process group for tensor parallelism.
    
    This sets up the distributed environment needed for SGLang's
    tensor parallelism using the sglang_backend distributed utilities.
    For TP > 1, this should be launched via torchrun.
    
    Returns the local rank (tp_rank).
    """
    from ..sglang_backend.distributed import init_sglang_distributed
    
    tp_size = config.tp_size
    timeout = config.sglang_backend_args.dist_timeout if config.sglang_backend_args else 20
    
    return init_sglang_distributed(tp_size=tp_size, timeout=timeout)


def run_worker(config: InferenceWorkerConfig) -> None:
    """
    Run a single inference worker (blocking).
    
    For TP > 1, this should be launched via torchrun:
        torchrun --standalone --nproc_per_node=4 -m specforge.modeling.target.remote_backend ...
    
    Only rank 0 handles ZMQ communication (task pulling, notifications).
    All ranks participate in inference.
    """
    tp_rank = _init_distributed_for_tp(config)
    
    worker = InferenceWorker(config)

    from ..sglang_backend.distributed import destroy_sglang_distributed

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        worker.shutdown()
        destroy_sglang_distributed()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        worker.setup()
        worker.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        worker.shutdown()
        destroy_sglang_distributed()