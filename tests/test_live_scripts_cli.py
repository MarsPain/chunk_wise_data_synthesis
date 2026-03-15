import importlib.util
import unittest
from pathlib import Path

from path_setup import ensure_src_path

ensure_src_path()

from generation_types import GenerationConfig


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveScriptsCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls._rephrase_script = _load_module(
            "run_live_openai_pipeline",
            repo_root / "scripts" / "run_live_openai_pipeline.py",
        )
        cls._generation_script = _load_module(
            "run_live_openai_generation_pipeline",
            repo_root / "scripts" / "run_live_openai_generation_pipeline.py",
        )

    def test_rephrase_script_accepts_prompt_language(self) -> None:
        parser = self._rephrase_script.build_parser()
        args = parser.parse_args(["--prompt-language", "zh"])
        self.assertEqual(args.prompt_language, "zh")

    def test_generation_script_accepts_prompt_language(self) -> None:
        parser = self._generation_script.build_parser()
        args = parser.parse_args(["--prompt-language", "zh"])
        self.assertEqual(args.prompt_language, "zh")

    def test_generation_script_accepts_profile_and_switches(self) -> None:
        parser = self._generation_script.build_parser()
        args = parser.parse_args(
            [
                "--profile",
                "cost_first",
                "--prompt-compression",
                "off",
                "--section-retry-strategy",
                "aggressive",
                "--consistency-pass",
                "on",
                "--consistency-guard",
                "off",
            ]
        )
        self.assertEqual(args.profile, "cost_first")
        self.assertEqual(args.prompt_compression, "off")
        self.assertEqual(args.section_retry_strategy, "aggressive")
        self.assertEqual(args.consistency_pass, "on")
        self.assertEqual(args.consistency_guard, "off")

    def test_generation_script_default_profile_is_coherence_first(self) -> None:
        parser = self._generation_script.build_parser()
        args = parser.parse_args([])
        config, snapshot = self._generation_script.resolve_generation_config(args)
        self.assertIsInstance(config, GenerationConfig)
        self.assertEqual(args.profile, "coherence_first")
        self.assertEqual(snapshot["profile"], "coherence_first")
        self.assertEqual(config.prompt_compression_enabled, False)
        self.assertEqual(config.max_section_retries, 3)
        self.assertEqual(config.consistency_pass_enabled, True)
        self.assertEqual(config.consistency_guard_enabled, True)

    def test_generation_script_cost_first_profile_prefers_low_cost(self) -> None:
        parser = self._generation_script.build_parser()
        args = parser.parse_args(["--profile", "cost_first"])
        config, snapshot = self._generation_script.resolve_generation_config(args)
        self.assertEqual(snapshot["profile"], "cost_first")
        self.assertEqual(config.prompt_compression_enabled, True)
        self.assertEqual(config.max_section_retries, 1)
        self.assertEqual(config.consistency_pass_enabled, False)
        self.assertEqual(config.consistency_guard_enabled, False)

    def test_generation_script_explicit_switch_overrides_profile(self) -> None:
        parser = self._generation_script.build_parser()
        args = parser.parse_args(
            [
                "--profile",
                "cost_first",
                "--consistency-pass",
                "on",
                "--consistency-guard",
                "on",
                "--prompt-compression",
                "off",
                "--section-retry-strategy",
                "balanced",
            ]
        )
        config, _ = self._generation_script.resolve_generation_config(args)
        self.assertEqual(config.consistency_pass_enabled, True)
        self.assertEqual(config.consistency_guard_enabled, True)
        self.assertEqual(config.prompt_compression_enabled, False)
        self.assertEqual(config.max_section_retries, 2)
        self.assertEqual(config.retry_on_missing_entities, True)


if __name__ == "__main__":
    unittest.main()
