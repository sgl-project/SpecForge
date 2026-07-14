# coding=utf-8
"""Qwen2.5-VL stays on the canonical PromptTask -> rollout -> Trainer path."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from specforge.config import Config
from specforge.data.utils import DataCollatorWithPadding
from specforge.data.vlm import QwenVLInputPreparer
from specforge.inference.adapters.policy import (
    EAGLE3_VLM_FEATURE_SCHEMA,
    PolicyFeatureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.inference.media import MediaInputs, PreparedTargetInput
from specforge.inference.target_engine import Eagle3TargetOutput
from specforge.runtime.contracts import PromptTask


class _CharacterTokenizer:
    """Small deterministic tokenizer that preserves real chat-template offsets."""

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        turns = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                pieces = []
                for item in content:
                    if item["type"] == "image":
                        pieces.append("<|vision_start|><|image_pad|><|vision_end|>")
                    else:
                        pieces.append(item["text"])
                content = "".join(pieces)
            turns.append(f'<|im_start|>{message["role"]}\n{content}<|im_end|>\n')
        return "".join(turns)

    @staticmethod
    def decode(input_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(chr(int(token)) for token in input_ids)


class _OffsetPreservingQwenProcessor:
    """Processor fixture that exercises the production VLM preprocessing code."""

    def __init__(self):
        self.tokenizer = _CharacterTokenizer()
        self.last_text = ""

    def __call__(
        self,
        *,
        text,
        images,
        videos,
        max_length,
        truncation,
        return_tensors,
        return_offsets_mapping,
        add_special_tokens,
    ):
        self.last_text = text[0][:max_length]
        assert images == ["decoded-image"]
        assert videos is None
        assert truncation and return_tensors == "pt" and return_offsets_mapping
        assert add_special_tokens is False
        ids = torch.tensor(
            [[ord(character) for character in self.last_text]], dtype=torch.long
        )
        offsets = torch.tensor(
            [[[index, index + 1] for index in range(len(self.last_text))]],
            dtype=torch.long,
        )
        return SimpleNamespace(
            input_ids=ids,
            offset_mapping=offsets,
            pixel_values=torch.arange(8, dtype=torch.float32).reshape(2, 4),
            image_grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.long),
        )


class _VLMPreparer:
    def prepare(self, task, device):
        length = len(task.payload["input_ids"])
        ids = torch.tensor(task.payload["input_ids"], device=device)
        return PreparedTargetInput(
            input_ids=ids,
            attention_mask=torch.ones(length, dtype=torch.long, device=device),
            loss_mask=torch.tensor(
                task.payload.get("loss_mask", [1] * length),
                dtype=torch.long,
                device=device,
            ),
            media=MediaInputs(
                pixel_values=torch.ones(4, 8, device=device),
                image_grid_thw=(torch.tensor([[1, 2, 2]], device=device),),
            ),
        )


class _VLMTarget:
    capture_layers = [1, 2, 3]

    def __init__(self):
        self.media_inputs = None

    def capture(self, input_ids, attention_mask, loss_mask, *, media_inputs):
        self.media_inputs = media_inputs
        batch, length = input_ids.shape
        return Eagle3TargetOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask.unsqueeze(-1),
            hidden_states=torch.zeros(batch, length, 12),
            target=torch.zeros(batch, length, 16),
        )

    def get_rope_index(self, *, input_ids, image_grid_thw, attention_mask):
        length = input_ids.shape[1]
        return torch.arange(length).view(1, 1, length).expand(3, 1, length), None


class _TrainablePathVLMTarget:
    """Network-free target that emits shape-real EAGLE3 VLM captures."""

    backend = "sglang"
    capture_layers = [1, 2, 3]

    def __init__(self, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.saw_media = False

    def capture(self, input_ids, attention_mask, loss_mask, *, media_inputs):
        self.saw_media = isinstance(media_inputs, MediaInputs)
        batch, length = input_ids.shape
        hidden = torch.arange(
            batch * length * 3 * self.hidden_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).reshape(batch, length, 3 * self.hidden_size)
        hidden = (hidden / hidden.numel()).to(torch.bfloat16)
        target = torch.zeros(
            batch,
            length,
            self.vocab_size,
            dtype=torch.bfloat16,
            device=input_ids.device,
        )
        target.scatter_(-1, input_ids.unsqueeze(-1), 8.0)
        return Eagle3TargetOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask.unsqueeze(-1),
            hidden_states=hidden,
            target=target,
        )

    def get_rope_index(self, *, input_ids, image_grid_thw, attention_mask):
        del image_grid_thw, attention_mask
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).view(1, 1, length)
        positions = positions.expand(3, batch, length).clone()
        if length >= 4:
            positions[1, :, 1:4] = torch.tensor([1, 1, 2], device=input_ids.device)
            positions[2, :, 1:4] = torch.tensor([1, 2, 1], device=input_ids.device)
        return positions, None


class UnifiedVLMPathTest(unittest.TestCase):
    @staticmethod
    def _config(*, target_backend="hf", batch_size=2, input_modality="qwen2_5_vl"):
        return Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                    "vocab_mapping_path": "/mapping.pt",
                    "target_backend": target_backend,
                    "input_modality": input_modality,
                },
                "data": {
                    "train_data_path": "/train.jsonl",
                    "chat_template": "qwen2-vl",
                },
                "training": {"batch_size": batch_size, "max_steps": 1},
            }
        )

    def _capture(self):
        return CaptureConfig.from_strategy(
            required_features=EAGLE3_VLM_FEATURE_SCHEMA.names,
            aux_hidden_state_layer_ids=(1, 2, 3),
            target_repr="logits",
            target_hidden_size=4,
            target_vocab_size=16,
        )

    def test_media_is_ephemeral_and_only_mrope_is_stored(self):
        target = _VLMTarget()
        adapter = PolicyFeatureAdapter(
            target,
            schema=EAGLE3_VLM_FEATURE_SCHEMA,
            device="cpu",
            input_preparer=_VLMPreparer(),
        )
        task = PromptTask(
            "vlm-0",
            "run",
            "eagle3",
            {
                "input_ids": [1, 2, 3, 4],
                "loss_mask": [1, 1, 1, 1],
                "media": {"image": "image.jpg", "conversations": []},
            },
            4,
        )

        features = adapter.generate_features([task], capture=self._capture())[0]

        self.assertIsInstance(target.media_inputs, MediaInputs)
        self.assertEqual(features["position_ids"].shape, (3, 1, 4))
        self.assertNotIn("pixel_values", features)
        self.assertNotIn("image_grid_thw", features)
        self.assertEqual(
            set(features) - {"__aux_layer_ids__"},
            set(EAGLE3_VLM_FEATURE_SCHEMA.names),
        )

    def test_unified_assembly_selects_padding_only_for_vlm_online(self):
        from specforge.training.assembly import _online_collate_override

        self.assertIsInstance(
            _online_collate_override(self._config()), DataCollatorWithPadding
        )
        self.assertIsNone(_online_collate_override(self._config(input_modality="text")))

    def test_vlm_collator_pads_ragged_features_and_three_axis_mrope(self):
        from specforge.training.assembly import _online_collate_override

        def sample(length, offset):
            return {
                "input_ids": torch.arange(length).view(1, length),
                "attention_mask": torch.ones(1, length, dtype=torch.long),
                "loss_mask": torch.ones(1, length, 1, dtype=torch.long),
                "hidden_state": torch.zeros(1, length, 12),
                "target": torch.zeros(1, length, 16),
                "position_ids": (
                    torch.arange(length).view(1, 1, length).expand(3, 1, length)
                    + offset
                ),
            }

        collator = _online_collate_override(self._config())
        batch = collator([sample(3, 0), sample(5, 10)])
        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["attention_mask"].shape, (2, 5))
        self.assertEqual(batch["loss_mask"].shape, (2, 5, 1))
        self.assertEqual(batch["hidden_state"].shape, (2, 5, 12))
        self.assertEqual(batch["target"].shape, (2, 5, 16))
        self.assertEqual(batch["position_ids"].shape, (3, 2, 5))
        self.assertTrue(torch.equal(batch["input_ids"][0, 3:], torch.zeros(2)))
        self.assertTrue(torch.equal(batch["attention_mask"][0, 3:], torch.zeros(2)))
        self.assertTrue(torch.equal(batch["loss_mask"][0, 3:], torch.zeros(2, 1)))
        self.assertTrue(torch.equal(batch["position_ids"][:, 0, 3:], torch.zeros(3, 2)))
        self.assertEqual(int(batch["position_ids"][0, 1, 0]), 10)

    def test_custom_vlm_target_uses_observable_hf_compatibility_backend(self):
        from specforge.training.assembly import _build_target_engine
        from specforge.training.strategies.registry import resolve_strategy

        target = mock.Mock()
        target.backend = "hf"
        cfg = self._config(target_backend="custom")
        with (
            mock.patch(
                "specforge.inference.target_engine.get_target_engine",
                return_value=target,
            ) as get_engine,
            mock.patch("specforge.training.assembly._device", return_value="cpu"),
            self.assertWarnsRegex(
                UserWarning, "compatibility alias.*effective target backend 'hf'"
            ),
        ):
            result = _build_target_engine(cfg, [1, 2, 3], resolve_strategy("eagle3"))

        self.assertIs(result, target)
        self.assertEqual(result.backend, "hf")
        self.assertEqual(get_engine.call_args.kwargs["backend"], "hf")
        self.assertEqual(get_engine.call_args.kwargs["input_modality"], "qwen2_5_vl")
        target.set_capture_layers.assert_called_once_with([1, 2, 3])

    def test_build_training_run_passes_vlm_padding_collator(self):
        from specforge.training.assembly import ModelBundle, build_training_run

        cfg = self._config(batch_size=2)
        bundle = ModelBundle(
            model=object(),
            draft_model=object(),
            tokenizer=object(),
            processor=object(),
            input_preparer=object(),
            feature_schema=EAGLE3_VLM_FEATURE_SCHEMA,
            target_engine=object(),
            target_hidden_size=8,
            target_vocab_size=16,
            draft_vocab_size=16,
            capture_layers=[1, 2, 3],
            strategy_kwargs={},
        )
        prompts = [
            {
                "task_id": f"vlm-{index}",
                "payload": {"input_ids": [index + 1], "loss_mask": [1]},
            }
            for index in range(2)
        ]
        trainer = object()
        with (
            mock.patch(
                "specforge.training.assembly.build_model_bundle",
                return_value=bundle,
            ),
            mock.patch(
                "specforge.training.assembly._prepare_prompts",
                return_value=prompts,
            ),
            mock.patch("specforge.training.assembly._device", return_value="cpu"),
            mock.patch(
                "specforge.launch.build_online_runtime", return_value=trainer
            ) as build,
        ):
            run = build_training_run(cfg)

        self.assertIs(run.trainer, trainer)
        self.assertIsInstance(
            build.call_args.kwargs["collate_fn"], DataCollatorWithPadding
        )

    def test_raw_qwen_vl_loss_mask_and_media_survive_canonical_ingest(self):
        """Use the real preprocessor, prompt builder, and rollout preparer.

        The processor fixture is offset-preserving and network-free; only image
        decoding is stubbed. This catches the deleted VLM gate's important
        contract: user/image tokens stay masked, assistant tokens train, and
        media tensors are re-materialized without entering ``PromptTask``.
        """

        from specforge.data.prompt_builder import prepare_prompt_tasks

        processor = _OffsetPreservingQwenProcessor()
        conversation = [
            {"role": "user", "content": "what is in the image?"},
            {"role": "assistant", "content": "This is an image of a cat."},
        ]
        record = {"image": "fixture.png", "conversations": conversation}

        with tempfile.TemporaryDirectory(prefix="qwen_vl_prompt_") as directory:
            data_path = Path(directory) / "vlm.jsonl"
            data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with (
                mock.patch("specforge.data.preprocessing.HAS_QWEN_VL_UTILS", True),
                mock.patch(
                    "specforge.data.preprocessing.process_vision_info",
                    return_value=(["decoded-image"], None),
                ) as process_vision,
            ):
                prompts = prepare_prompt_tasks(
                    data_path,
                    processor.tokenizer,
                    chat_template="qwen2-vl",
                    max_length=512,
                    is_preformatted=False,
                    train_only_last_turn=False,
                    cache_dir=None,
                    cache_key=None,
                    num_proc=1,
                    input_modality="qwen2_5_vl",
                    processor=processor,
                )
                self.assertEqual(len(prompts), 1)
                payload = prompts[0]["payload"]
                self.assertEqual(
                    payload["media"],
                    {"image": "fixture.png", "conversations": conversation},
                )
                self.assertNotIn("pixel_values", payload)
                self.assertNotIn("image_grid_thw", payload)

                decoded = processor.tokenizer.decode(
                    payload["input_ids"], skip_special_tokens=False
                )
                loss_mask = payload["loss_mask"]
                user_start = decoded.index("what is in the image?")
                user_end = user_start + len("what is in the image?")
                answer_start = decoded.index("This is an image of a cat.")
                answer_end = answer_start + len("This is an image of a cat.")
                self.assertEqual(set(loss_mask[user_start:user_end]), {0})
                self.assertEqual(set(loss_mask[answer_start:answer_end]), {1})

                task = PromptTask(
                    "vlm-real-0",
                    "run",
                    "eagle3",
                    payload,
                    512,
                )
                prepared = QwenVLInputPreparer(processor, "qwen2-vl").prepare(
                    task, "cpu"
                )

        self.assertEqual(process_vision.call_count, 2)
        self.assertTrue(
            torch.equal(
                prepared.input_ids,
                torch.tensor(payload["input_ids"], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.loss_mask,
                torch.tensor(payload["loss_mask"], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.media.pixel_values,
                torch.arange(8, dtype=torch.float32).reshape(2, 4),
            )
        )
        self.assertEqual(
            tuple(prepared.media.image_grid_thw[0].shape),
            (1, 3),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "VLM optimizer smoke needs CUDA")
    def test_canonical_vlm_batch_two_runs_mrope_backward_and_optimizer_step(self):
        """Exercise ragged VLM rollout -> padded batch -> M-RoPE training."""

        from specforge.core.eagle3 import OnlineEagle3Model
        from specforge.launch import build_online_runtime
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.optimizer import BF16Optimizer
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29583")
        hidden_size, vocab_size = 32, 64
        lengths = (8, 6)
        with tempfile.TemporaryDirectory(prefix="vlm_train_step_") as directory:
            draft_config = {
                **fx.TINY_DRAFT_CONFIG,
                "hidden_size": hidden_size,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": vocab_size,
                "draft_vocab_size": vocab_size,
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [1, 1, 2],
                },
                "target_model_type": "qwen2_5_vl",
            }
            config_path = Path(directory) / "draft.json"
            config_path.write_text(json.dumps(draft_config), encoding="utf-8")
            draft = AutoDraftModel.from_config(
                AutoDraftModelConfig.from_file(str(config_path)),
                attention_backend="flex_attention",
                torch_dtype=torch.bfloat16,
            ).cuda()
            draft.freeze_embedding()
            self.assertEqual(
                type(draft.midlayer.self_attn.rotary_emb).__name__,
                "LlamaMutiRotaryEmbedding",
            )
            eagle = OnlineEagle3Model(
                draft,
                length=2,
                attention_backend="flex_attention",
            ).cuda()
            target = _TrainablePathVLMTarget(hidden_size, vocab_size)
            losses = []
            prompts = [
                {
                    "payload": {
                        "input_ids": list(range(1, length + 1)),
                        "loss_mask": [0, 0] + [1] * (length - 2),
                        "media": {
                            "image": f"fixture-{index}.png",
                            "conversations": [],
                        },
                    }
                }
                for index, length in enumerate(lengths)
            ]
            trainer = build_online_runtime(
                strategy="eagle3",
                target_model=target,
                prompts=prompts,
                draft_model=eagle,
                optimizer_factory=lambda model: BF16Optimizer(
                    model,
                    lr=1e-3,
                    max_grad_norm=0.5,
                    warmup_ratio=0.0,
                    total_steps=1,
                ),
                run_id="vlm-train-step",
                output_dir=str(Path(directory) / "output"),
                target_hidden_size=hidden_size,
                target_vocab_size=vocab_size,
                draft_vocab_size=vocab_size,
                target_repr="logits",
                aux_hidden_state_layer_ids=(1, 2, 3),
                batch_size=2,
                max_steps=1,
                feature_schema=EAGLE3_VLM_FEATURE_SCHEMA,
                input_preparer=_VLMPreparer(),
                collate_fn=DataCollatorWithPadding(),
                logger=lambda metrics, _step: losses.append(metrics["loss"]),
                log_interval=1,
            )

            self.assertEqual(trainer.fit(), 1)

        self.assertTrue(target.saw_media)
        self.assertEqual(len(losses), 1)
        self.assertTrue(torch.isfinite(torch.tensor(losses[0])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
