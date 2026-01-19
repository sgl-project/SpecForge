from typing import Any, Callable, Dict, Iterator, List, Optional

from torch.utils.data import DataLoader


class PrefetchingDataLoader:
    """
    DataLoader wrapper that prefetches inference tasks ahead of time for remote backends.
    
    Inference runs asynchronously - the dataloader maintains a queue of pending tasks
    and refills it aggressively to keep inference workers busy. When the queue drops
    below a threshold, it submits multiple tasks at once.
    
    Usage:
        dataloader = PrefetchingDataLoader(
            base_dataloader,
            target_model=remote_model,
            prefetch_depth=8,
            low_watermark=4,
            refill_batch_size=10,
        )
        
        for data, eagle3_data in dataloader:
            # data: the original batch from the dataloader
            # eagle3_data: the pre-computed inference result
            plosses, acces = run_forward(args, eagle3_model, data, eagle3_data=eagle3_data)
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        target_model,
        prefetch_depth: int = 8,
        low_watermark: int = 4,
        refill_batch_size: int = 10,
        submit_fn: Optional[Callable] = None,
        get_result_fn: Optional[Callable] = None,
    ):
        """
        Args:
            dataloader: The base DataLoader to wrap
            target_model: The remote target model with submit_task/get_result methods
            prefetch_depth: Target number of pending tasks to maintain
            low_watermark: When queue drops to this level, trigger refill
            refill_batch_size: Number of tasks to submit at once during refill
            submit_fn: Optional custom function to submit a task (default: target_model.submit_task)
            get_result_fn: Optional custom function to get a result (default: target_model.get_result)
        """
        self.dataloader = dataloader
        self.target_model = target_model
        self.prefetch_depth = prefetch_depth
        self.low_watermark = low_watermark
        self.refill_batch_size = refill_batch_size
        
        self._submit_fn = submit_fn or self._default_submit
        self._get_result_fn = get_result_fn or self._default_get_result
        
        self._pending_task_ids: List[str] = []
        self._pending_data: List[Dict] = []
        self._data_iter: Optional[Iterator] = None
        self._exhausted: bool = False
    
    def _default_submit(self, data: Dict) -> str:
        return self.target_model.submit_task(
            input_ids=data["input_ids"].cuda(),
            attention_mask=data["attention_mask"].cuda(),
            loss_mask=data["loss_mask"].cuda(),
        )
    
    def _default_get_result(self, task_id: str):
        return self.target_model.get_result(task_id)
    
    def _refill_queue(self) -> None:
        """Submit tasks to keep the queue at prefetch_depth."""
        if self._exhausted:
            return
        
        tasks_to_submit = min(
            self.refill_batch_size,
            self.prefetch_depth - len(self._pending_task_ids),
        )
        
        for _ in range(tasks_to_submit):
            try:
                data = next(self._data_iter)
                task_id = self._submit_fn(data)
                self._pending_task_ids.append(task_id)
                self._pending_data.append(data)
            except StopIteration:
                self._exhausted = True
                break
    
    def __iter__(self) -> Iterator:
        self._pending_task_ids = []
        self._pending_data = []
        self._data_iter = iter(self.dataloader)
        self._exhausted = False
        
        self._refill_queue()
        
        while self._pending_task_ids:
            if len(self._pending_task_ids) <= self.low_watermark:
                self._refill_queue()
            
            oldest_task_id = self._pending_task_ids.pop(0)
            oldest_data = self._pending_data.pop(0)
            eagle3_data = self._get_result_fn(oldest_task_id)
            yield oldest_data, eagle3_data
    
    def __len__(self) -> int:
        return len(self.dataloader)
    
    @property
    def sampler(self):
        return self.dataloader.sampler
    
    @property
    def dataset(self):
        return self.dataloader.dataset
    
    @property
    def pending_count(self) -> int:
        return len(self._pending_task_ids)
