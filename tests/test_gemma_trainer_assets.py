import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "skills" / "gemma-trainer" / "assets"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("distill_dataset", ASSETS / "distill_dataset.py")

    def make_args(self):
        return types.SimpleNamespace(
            input="unused.json",
            use_ollama=False,
            model="teacher-model",
            force_no_qlora=False,
            max_new_tokens=32,
            temperature=0.25,
            system_prompt="Be helpful.",
        )

    def test_hf_response_distillation_passes_quantization_choice(self):
        args = self.make_args()
        pipeline = types.SimpleNamespace(tokenizer=object())
        config = types.SimpleNamespace()
        fake_transformers = types.SimpleNamespace(TextStreamer=lambda tokenizer: object())

        with (
            mock.patch.object(self.module, "load_seed_prompts", return_value=["Hello"]),
            mock.patch.object(self.module, "init_hf_pipeline", return_value=(pipeline, config)) as init,
            mock.patch.object(self.module, "generate_response_hf", return_value="Hi"),
            mock.patch.dict(sys.modules, {"transformers": fake_transformers}),
        ):
            self.module.run_response_distillation(args)

        init.assert_called_once_with("teacher-model", True)

    def test_hf_response_distillation_applies_temperature_and_assistant_role(self):
        args = self.make_args()
        pipeline = types.SimpleNamespace(tokenizer=object())
        config = types.SimpleNamespace()
        fake_transformers = types.SimpleNamespace(TextStreamer=lambda tokenizer: object())

        with (
            mock.patch.object(self.module, "load_seed_prompts", return_value=["Hello"]),
            mock.patch.object(self.module, "init_hf_pipeline", return_value=(pipeline, config)),
            mock.patch.object(self.module, "generate_response_hf", return_value="Hi"),
            mock.patch.dict(sys.modules, {"transformers": fake_transformers}),
        ):
            dataset = self.module.run_response_distillation(args)

        self.assertEqual(config.temperature, 0.25)
        self.assertEqual(dataset[0]["messages"][-1]["role"], "assistant")


class DatasetValidationTests(unittest.TestCase):
    def test_dataset_prep_imports_without_transformers(self):
        with mock.patch.dict(sys.modules, {"transformers": None}):
            module = load_module("dataset_prep_without_transformers", ASSETS / "dataset_prep.py")

        self.assertIsNone(module.AutoTokenizer)

    def test_sft_validation_works_without_a_tokenizer(self):
        fake_transformers = types.SimpleNamespace(AutoTokenizer=object())
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            module = load_module("dataset_prep", ASSETS / "dataset_prep.py")

        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps([sample]), encoding="utf-8")
            self.assertTrue(
                module.run_validation(
                    str(path), "sft", max_seq_length=2048, tokenizer_name=None
                )
            )


class SkillDocumentationTests(unittest.TestCase):
    def test_skill_uses_standard_assistant_role_and_current_links(self):
        text = (ROOT / "skills" / "gemma-trainer" / "SKILL.md").read_text()
        self.assertIn('{"role": "assistant"', text)
        self.assertNotIn('{"role": "model"', text)
        self.assertNotIn("saving-to-gguf.md", text)
        self.assertNotIn("gemma-4.md.txt", text)


if __name__ == "__main__":
    unittest.main()
