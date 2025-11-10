from sglang.srt.model_executor.model_runner import ModelRunner

from .patch import (
    init_distributed_environment,
    initialize_dp_attention,
    initialize_model_parallel,
    wrap_eagle3_logits_processors_in_module,
)


class SGLangRunner(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wrap_eagle3_logits_processors_in_module(self.model)

    def init_torch_distributed(self):
        from unittest.mock import patch

        def do_nothing(*args, **kwargs):
            pass

        with patch(
            "sglang.srt.model_executor.model_runner.init_distributed_environment",
            init_distributed_environment,
        ), patch(
            "sglang.srt.model_executor.model_runner.initialize_model_parallel",
            initialize_model_parallel,
        ), patch(
            "sglang.srt.model_executor.model_runner.initialize_dp_attention",
            initialize_dp_attention,
        ), patch(
            "sglang.srt.model_executor.model_runner.log_info_on_rank0", do_nothing
        ):
            return super().init_torch_distributed()
