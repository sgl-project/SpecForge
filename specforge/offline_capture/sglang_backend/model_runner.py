import logging

from sglang.srt.distributed.parallel_state import (
    get_attn_tp_group,
    get_pp_group,
    get_tp_group,
)
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class SGLangRunner(ModelRunner):
    """Offline ModelRunner for the local capture backend (no scheduler).

    The parallel runtime itself is built by ``bootstrap.init_parallel_runtime``
    in the caller, mirroring the upstream offline benchmark entry; this
    subclass only republishes the group handles that capture code reads
    directly off the runner.
    """

    _last_forward_batch = None

    def init_torch_distributed(self):
        super().init_torch_distributed()
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attn_tp_group()
