import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "skills" / "gemma-trainer" / "assets"


def load_training_utils():
    path = ASSETS / "training_utils.py"
    spec = importlib.util.spec_from_file_location("training_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DpoFormattingTests(unittest.TestCase):
    def test_converts_preference_strings_without_repeating_prompt(self):
        utils = load_training_utils()

        result = utils.to_conversational_preference(
            {"prompt": "Question", "chosen": "Good", "rejected": "Bad"}
        )

        self.assertEqual(
            result,
            {
                "prompt": [{"role": "user", "content": "Question"}],
                "chosen": [{"role": "assistant", "content": "Good"}],
                "rejected": [{"role": "assistant", "content": "Bad"}],
            },
        )


class AssistantMaskTests(unittest.TestCase):
    def test_masks_every_non_assistant_span_in_multi_turn_tokens(self):
        utils = load_training_utils()
        tokens = [1, 2, 10, 11, 20, 21, 99, 3, 10, 11, 22, 99, 0]

        labels = utils.mask_assistant_tokens(
            tokens,
            assistant_start=[10, 11],
            turn_end=[99],
            pad_token_id=0,
        )

        self.assertEqual(
            labels,
            [-100, -100, -100, -100, 20, 21, 99, -100, -100, -100, 22, 99, -100],
        )

    def test_keeps_a_truncated_final_assistant_response(self):
        utils = load_training_utils()

        labels = utils.mask_assistant_tokens(
            [1, 10, 11, 20, 21],
            assistant_start=[10, 11],
            turn_end=[99],
            pad_token_id=0,
        )

        self.assertEqual(labels, [-100, -100, -100, 20, 21])

    def test_rejects_examples_without_an_assistant_turn(self):
        utils = load_training_utils()

        with self.assertRaisesRegex(ValueError, "assistant turn"):
            utils.mask_assistant_tokens(
                [1, 2, 3],
                assistant_start=[10, 11],
                turn_end=[99],
                pad_token_id=0,
            )


class ProcessorArgumentsTests(unittest.TestCase):
    def test_enables_truncation_at_the_training_max_length(self):
        utils = load_training_utils()

        kwargs = utils.processor_kwargs(["text"], max_length=4096)

        self.assertTrue(kwargs["truncation"])
        self.assertEqual(kwargs["max_length"], 4096)


class ScriptIntegrationTests(unittest.TestCase):
    def test_dpo_uses_conversational_preferences_and_explicit_beta(self):
        source = (ASSETS / "dpo_train.py").read_text()

        self.assertIn("raw_dataset.map(to_conversational_preference)", source)
        self.assertNotIn("def apply_dpo_template", source)
        self.assertIn("beta=0.1", source)

    def test_sft_uses_multimodal_unsloth_loader_and_safe_hf_collation(self):
        source = (ASSETS / "sft_train.py").read_text()

        self.assertIn("from unsloth import FastVisionModel", source)
        self.assertIn("UnslothVisionDataCollator", source)
        self.assertIn("get_collate_fn(processor, max_length)", source)
        self.assertIn("mask_assistant_tokens", source)


if __name__ == "__main__":
    unittest.main()
