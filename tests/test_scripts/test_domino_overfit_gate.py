import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    path = ROOT / "scripts" / "gates" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


create_data = load_script("select_overfit_sample.py")
check_overfit = load_script("check_overfit_metrics.py")


def fake_processing_stack(loss_tokens=40):
    class FakeScalar:
        def item(self):
            return loss_tokens

    class FakeMask:
        def sum(self):
            return FakeScalar()

    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            del add_special_tokens
            return list(text)

        def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
            prompt = "PROMPT:"
            if add_generation_prompt:
                return prompt
            assistant = messages[-1]
            return (
                prompt + assistant.get("reasoning_content", "") + assistant["content"]
            )

    def preprocess(
        _tokenizer,
        _conversations,
        _template,
        *,
        max_length,
    ):
        del max_length
        return {"loss_mask": [FakeMask()]}

    return FakeTokenizer(), object(), preprocess


class TestCreateDominoOverfitData(unittest.TestCase):
    def test_selects_clean_single_turn_non_reasoning_sample(self):
        rows = [
            {
                "id": "multi",
                "conversations": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a" * 200},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "b" * 200},
                ],
            },
            {
                "id": "thinking",
                "conversations": [
                    {"role": "user", "content": "u"},
                    {
                        "role": "assistant",
                        "content": "clean" * 40,
                        "reasoning_content": "hidden",
                    },
                ],
            },
            {
                "id": "selected",
                "extra": "not copied",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer " * 30},
                ],
                "status": "success",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.jsonl")
            output = os.path.join(tmp, "one.jsonl")
            with open(source, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            index, selected, loss_tokens = create_data.create_single_sample(
                source,
                output,
                model_path="unused",
                reasoning_policy="forbidden",
                processing_stack=fake_processing_stack(),
            )
            self.assertEqual(index, 2)
            self.assertEqual(selected["id"], "selected")
            self.assertEqual(loss_tokens, 40)
            self.assertNotIn("extra", selected)
            with open(output, encoding="utf-8") as handle:
                written = [json.loads(line) for line in handle]
            self.assertEqual(written, [selected])

    def test_rejects_empty_assistant_and_existing_output(self):
        row = {
            "conversations": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": ""},
            ]
        }
        self.assertIsNone(create_data.single_turn_candidate(row))
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.jsonl")
            output = os.path.join(tmp, "one.jsonl")
            Path(source).write_text("{}\n")
            Path(output).write_text("keep\n")
            with self.assertRaises(FileExistsError):
                create_data.create_single_sample(
                    source,
                    output,
                    model_path="unused",
                    processing_stack=fake_processing_stack(),
                )
            self.assertEqual(Path(output).read_text(), "keep\n")

    def test_skips_candidate_below_real_loss_token_contract(self):
        rows = [
            {
                "id": "too-few-loss-tokens",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer " * 30},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.jsonl")
            output = os.path.join(tmp, "one.jsonl")
            Path(source).write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "at least 32.*loss tokens"):
                create_data.create_single_sample(
                    source,
                    output,
                    model_path="unused",
                    min_loss_tokens=32,
                    processing_stack=fake_processing_stack(loss_tokens=31),
                )
            self.assertFalse(os.path.exists(output))

    def test_required_reasoning_is_preserved_in_prompt_artifact(self):
        row = {
            "id": "reasoning-profile",
            "conversations": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "reasoning_content": "reasoning ",
                    "content": "answer " * 30,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.jsonl")
            output = os.path.join(tmp, "one.jsonl")
            artifact = os.path.join(tmp, "prompt.json")
            Path(source).write_text(json.dumps(row) + "\n")
            create_data.create_single_sample(
                source,
                output,
                model_path="unused",
                reasoning_policy="required",
                prompt_output_path=artifact,
                enable_thinking=True,
                processing_stack=fake_processing_stack(),
            )
            selected = json.loads(Path(output).read_text())
            prompt = json.loads(Path(artifact).read_text())
            self.assertEqual(
                selected["conversations"][-1]["reasoning_content"], "reasoning "
            )
            self.assertEqual(prompt["target_suffix"], "reasoning " + "answer " * 30)
            self.assertTrue(prompt["enable_thinking"])
            self.assertNotIn("reasoning_content", prompt["prompt_messages"][0])

    def test_untruncated_gate_skips_oversized_candidate(self):
        rows = [
            {
                "id": "truncated",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "a" * 200},
                ],
            },
            {
                "id": "selected",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "b" * 128},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            output = Path(tmp) / "one.jsonl"
            artifact = Path(tmp) / "prompt.json"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            index, selected, _ = create_data.create_single_sample(
                str(source),
                str(output),
                model_path="unused",
                max_length=150,
                prompt_output_path=str(artifact),
                require_untruncated=True,
                processing_stack=fake_processing_stack(),
            )
            self.assertEqual(index, 1)
            self.assertEqual(selected["id"], "selected")
            self.assertLessEqual(
                json.loads(artifact.read_text())["full_input_tokens"], 150
            )


class TestDominoOverfitGate(unittest.TestCase):
    def test_passes_on_final_zero_loss_full_accuracy_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "train.log")
            Path(log).write_text(
                "[consumer] step 1 {'loss': 1.2, 'accuracy': 0.2}\n"
                "[consumer] step 10 {'loss': 0.0, 'accuracy': 1.0}\n"
            )
            checkpoint = Path(tmp) / "consumer" / "run-step10"
            checkpoint.mkdir(parents=True)
            (checkpoint / "training_state.pt").touch()
            (checkpoint / "_SUCCESS").touch()
            result = check_overfit.check_overfit(
                log,
                str(Path(tmp) / "consumer"),
                expected_step=10,
                max_loss=1e-4,
                min_accuracy=1.0,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["checkpoint"], str(checkpoint / "training_state.pt")
            )

    def test_accepts_unified_trainer_log_and_selects_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text(
                "step 2: {'loss': 0.2, 'accuracy': 0.8}\n"
                "step 10: {'loss': 0.0, 'accuracy': 1.0}\n",
                encoding="utf-8",
            )
            checkpoint_root = Path(tmp) / "checkpoints"
            for step in (2, 10):
                checkpoint = checkpoint_root / f"run-step{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "training_state.pt").touch()
                (checkpoint / "_SUCCESS").touch()

            result = check_overfit.check_overfit(
                str(log),
                str(checkpoint_root),
                expected_step=10,
                max_loss=1e-4,
                min_accuracy=1.0,
            )

            self.assertTrue(result["passed"])
            self.assertEqual(
                result["checkpoint"],
                str(checkpoint_root / "run-step10" / "training_state.pt"),
            )

    def test_ignores_markerless_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text("step 10: {'loss': 0.0, 'accuracy': 1.0}\n")
            checkpoint = Path(tmp) / "checkpoints" / "run-step10"
            checkpoint.mkdir(parents=True)
            (checkpoint / "training_state.pt").touch()

            with self.assertRaisesRegex(ValueError, "no committed checkpoint"):
                check_overfit.check_overfit(
                    str(log),
                    str(checkpoint.parent),
                    expected_step=10,
                    max_loss=1e-4,
                    min_accuracy=1.0,
                )

    def test_fails_when_any_gate_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "train.log")
            Path(log).write_text(
                "[consumer] step 9 {'loss': 0.01, 'accuracy': 0.9999}\n"
            )
            with self.assertRaisesRegex(
                ValueError, "final logged step.*final loss.*token accuracy.*checkpoint"
            ):
                check_overfit.check_overfit(
                    log,
                    os.path.join(tmp, "consumer"),
                    expected_step=10,
                    max_loss=1e-4,
                    min_accuracy=1.0,
                )


if __name__ == "__main__":
    unittest.main()
