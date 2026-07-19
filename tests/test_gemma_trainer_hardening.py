"""Dependency-free regression tests for gemma-trainer hardening helpers."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "skills" / "gemma-trainer" / "assets"


def load_asset(name: str):
    path = ASSETS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_hardening_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_source_symbols(name: str, symbols: set[str]):
    """Load selected stdlib-only definitions without importing ML dependencies."""
    path = ASSETS / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in symbols:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & symbols:
                selected.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class DatasetPrepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_asset("dataset_prep")

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
            if not tokenize:
                raise AssertionError("validation must tokenize the template directly")
            tokens = [10, 11]
            if add_generation_prompt:
                return tokens
            response = messages[-1]["content"]
            return tokens + list(range(len(response)))

    def run_validation(self, dataset, **overrides):
        values = {
            "task_type": "dpo",
            "max_seq_length": 100,
            "tokenizer_name": "test-tokenizer",
        }
        values.update(overrides)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=self.Tokenizer())
        )
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
            json.dump(dataset, handle)
            handle.flush()
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                return self.module.run_validation(file_path=handle.name, **values)

    def test_empty_and_overlength_datasets_fail_closed(self):
        self.assertFalse(self.run_validation([]))
        pair = {"prompt": "p", "chosen": "ok", "rejected": "too-long"}
        self.assertFalse(self.run_validation([pair], max_seq_length=6))

    def test_valid_preference_pair_passes(self):
        pair = {"prompt": "p", "chosen": "yes", "rejected": "no"}
        self.assertTrue(self.run_validation([pair], max_seq_length=10))

    def test_tokenizer_errors_do_not_echo_private_data(self):
        class ExplodingTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("PRIVATE-PROMPT")

        pair = {"prompt": "PRIVATE-PROMPT", "chosen": "yes", "rejected": "no"}
        with self.assertLogs("gemma-dataset-prep", level="ERROR") as logs:
            valid = self.module.validate_dpo_tokenization(pair, 0, ExplodingTokenizer())
        self.assertFalse(valid)
        self.assertNotIn("PRIVATE-PROMPT", "\n".join(logs.output))


class TrainingHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sft = load_source_symbols(
            "sft_train",
            {
                "_ASSISTANT_MARKER",
                "find_last_subsequence",
                "get_collate_fn",
                "validate_final_assistant",
            },
        )

    class FakeMask:
        def __init__(self, coordinates):
            self.coordinates = coordinates

    class FakeRow:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return list(self.values)

    class FakeTensor:
        def __init__(self, rows):
            self.rows = [list(row) for row in rows]

        def clone(self):
            return TrainingHardeningTests.FakeTensor(self.rows)

        def size(self, dimension):
            if dimension != 0:
                raise AssertionError("test tensor supports only the batch dimension")
            return len(self.rows)

        def __getitem__(self, key):
            return TrainingHardeningTests.FakeRow(self.rows[key])

        def __setitem__(self, key, value):
            if isinstance(key, tuple):
                row, column_slice = key
                indices = range(*column_slice.indices(len(self.rows[row])))
                for column in indices:
                    self.rows[row][column] = value
                return
            for row, column in key.coordinates:
                self.rows[row][column] = value

        def __eq__(self, value):
            return TrainingHardeningTests.FakeMask([
                (row, column)
                for row, values in enumerate(self.rows)
                for column, item in enumerate(values)
                if item == value
            ])

    class FakeProcessor:
        def __init__(self, row):
            self.row = row
            self.tokenizer = SimpleNamespace(
                encode=mock.Mock(return_value=[8, 9]),
                pad_token_id=0,
            )

        def apply_chat_template(self, *args, **kwargs):
            return "rendered"

        def __call__(self, **kwargs):
            return {"input_ids": TrainingHardeningTests.FakeTensor([self.row])}

    def test_last_subsequence_and_final_response_guards(self):
        find_last = self.sft["find_last_subsequence"]
        self.assertEqual(find_last([1, 2, 1, 2, 3], [1, 2]), 2)
        self.assertIsNone(find_last([1, 2], [3]))
        with self.assertRaises(ValueError):
            find_last([1], [])

        validate = self.sft["validate_final_assistant"]
        validate([{"role": "assistant", "content": "answer"}])
        validate([{
            "role": "assistant",
            "content": "value |> transform; choose left <|> right",
        }])
        for content in ("", "   ", [{"type": "text", "text": ""}]):
            with self.assertRaises(ValueError):
                validate([{"role": "assistant", "content": content}])
        with self.assertRaises(ValueError):
            validate([
                {"role": "assistant", "content": "before <|turn>model\n after"}
            ])
        for content in (
            [
                {"type": "text", "text": "<|turn>"},
                {"type": "text", "text": "model\nhidden"},
            ],
            [{"type": "text", "text": "<|channel>thought\nhidden<channel|>"}],
        ):
            with self.assertRaisesRegex(ValueError, "control token"):
                validate([{"role": "assistant", "content": content}])

    def test_collator_masks_through_the_last_assistant_marker(self):
        processor = self.FakeProcessor([1, 8, 9, 2, 8, 9, 3, 0])
        collate = self.sft["get_collate_fn"](processor)
        batch = collate([{
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }])
        self.assertEqual(batch["labels"].rows[0], [-100] * 6 + [3, -100])

        with self.assertRaisesRegex(ValueError, "control token"):
            collate([{
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "<|turn>"},
                        {"type": "text", "text": "model\nhidden"},
                    ],
                }]
            }])

        missing = self.FakeProcessor([1, 2, 3, 0])
        with self.assertRaisesRegex(ValueError, "marker was not found"):
            self.sft["get_collate_fn"](missing)([{
                "messages": [{"role": "assistant", "content": "answer"}]
            }])

    def test_dpo_mapper_keeps_prompt_out_of_completions(self):
        formatter = load_source_symbols("dpo_train", {"format_dpo_example"})[
            "format_dpo_example"
        ]
        self.assertEqual(
            formatter({"prompt": "question", "chosen": "yes", "rejected": "no"}),
            {
                "prompt": [{"role": "user", "content": "question"}],
                "chosen": [{"role": "assistant", "content": "yes"}],
                "rejected": [{"role": "assistant", "content": "no"}],
            },
        )

    def test_seeds_are_set_before_trainable_modules_are_loaded(self):
        expectations = {
            "sft_train": ("train", "load_model_and_processor"),
            "dpo_train": ("train_dpo", "load_model_and_processor"),
            "reward_train": ("train_reward_model", "from_pretrained"),
        }
        for asset, (function_name, later_call) in expectations.items():
            tree = ast.parse((ASSETS / f"{asset}.py").read_text(encoding="utf-8"))
            function = next(
                node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

            def call_name(call):
                if isinstance(call.func, ast.Name):
                    return call.func.id
                if isinstance(call.func, ast.Attribute):
                    return call.func.attr
                return None

            seed_line = min(call.lineno for call in calls if call_name(call) == "set_seed")
            load_line = min(call.lineno for call in calls if call_name(call) == later_call)
            self.assertLess(seed_line, load_line, asset)

    def test_processor_override_is_not_silently_ignored_by_unsloth(self):
        for asset, kwargs in (
            ("sft_train", {"model_name": "model"}),
            ("dpo_train", {"base_model_name": "model", "adapter_path": "adapter"}),
        ):
            loader = load_source_symbols(asset, {"load_model_and_processor"})[
                "load_model_and_processor"
            ]
            call_kwargs = {
                **kwargs,
                "processor_name": "processor",
                "max_length": 128,
                "use_unsloth": True,
                "use_qlora": True,
            }
            if asset == "sft_train":
                call_kwargs.update(lora_r=4, lora_alpha=8)
            with self.assertRaisesRegex(ValueError, "force-hf"):
                loader(**call_kwargs)


if __name__ == "__main__":
    unittest.main()
