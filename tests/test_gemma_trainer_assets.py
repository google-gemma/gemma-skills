"""Dependency-free regression tests for the gemma-trainer asset scripts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "skills" / "gemma-trainer" / "assets"


def load_asset(name: str):
    path = ASSETS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_asset_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DistillDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_asset("distill_dataset")

    @staticmethod
    def response_args(**overrides):
        values = {
            "input": "unused.txt",
            "use_ollama": False,
            "ollama_url": "http://localhost:11434",
            "ollama_model": "gemma4:test",
            "model": "google/gemma-4-test",
            "system_prompt": "Be helpful.",
            "temperature": 0.35,
            "max_new_tokens": 77,
            "force_no_qlora": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_hf_response_uses_generation_settings_and_assistant_role(self):
        config = SimpleNamespace()
        captured = {}

        def generate(**kwargs):
            captured.update(kwargs)
            return "PRIVATE-RESPONSE"

        with (
            mock.patch.object(self.module, "load_seed_prompts", return_value=["PRIVATE-PROMPT"]),
            mock.patch.object(
                self.module, "init_hf_pipeline", return_value=("pipeline", config)
            ) as init_pipeline,
            mock.patch.object(self.module, "generate_response_hf", side_effect=generate),
            self.assertLogs("gemma-distill", level="INFO") as logs,
        ):
            result = self.module.run_response_distillation(self.response_args())

        init_pipeline.assert_called_once_with("google/gemma-4-test", True)
        self.assertEqual(config.max_new_tokens, 77)
        self.assertEqual(config.temperature, 0.35)
        self.assertFalse(hasattr(config, "temperatur"))
        self.assertEqual(captured["gen_kwargs"], {"generation_config": config})
        self.assertEqual(
            [message["role"] for message in result[0]["messages"]],
            ["system", "user", "assistant"],
        )
        rendered_logs = "\n".join(logs.output)
        self.assertNotIn("PRIVATE-PROMPT", rendered_logs)
        self.assertNotIn("PRIVATE-RESPONSE", rendered_logs)

    def test_ollama_response_uses_assistant_role(self):
        with (
            mock.patch.object(self.module, "load_seed_prompts", return_value=["question"]),
            mock.patch.object(self.module, "query_ollama_api", return_value="answer"),
        ):
            result = self.module.run_response_distillation(
                self.response_args(use_ollama=True)
            )

        self.assertEqual(
            [message["role"] for message in result[0]["messages"]],
            ["system", "user", "assistant"],
        )

    def test_hf_synthesis_uses_temperature_and_assistant_schema(self):
        config = SimpleNamespace()
        captured = {}
        response = json.dumps(
            [{"messages": [{"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"}]}]
        )

        def generate(**kwargs):
            captured.update(kwargs)
            return response

        args = self.response_args(input=None, num_samples=1)
        with (
            mock.patch.object(
                self.module, "init_hf_pipeline", return_value=("pipeline", config)
            ),
            mock.patch.object(self.module, "generate_response_hf", side_effect=generate),
        ):
            result = self.module.run_self_instruct_synthesis(args)

        self.assertEqual(config.temperature, 0.35)
        self.assertFalse(hasattr(config, "temperatur"))
        self.assertIn('"role": "assistant"', captured["prompt"])
        self.assertNotIn('"role": "model"', captured["prompt"])
        self.assertEqual(result, json.loads(response))


if __name__ == "__main__":
    unittest.main()
