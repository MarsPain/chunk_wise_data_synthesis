from pathlib import Path
import unittest


class LiveDataAssetTests(unittest.TestCase):
    def test_live_input_sample_exists_and_non_empty(self) -> None:
        sample_path = (
            Path(__file__).resolve().parent / "data" / "live_rephrase_input.txt"
        )
        self.assertTrue(sample_path.exists(), f"missing sample file: {sample_path}")
        content = sample_path.read_text(encoding="utf-8").strip()
        self.assertGreater(len(content), 80)


if __name__ == "__main__":
    unittest.main()
