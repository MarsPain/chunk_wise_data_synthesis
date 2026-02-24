import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
