from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class DocsValidationTests(unittest.TestCase):
    def test_docs_validator_passes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/validate_docs.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
            self.fail(f"docs validator failed with output:\n{output}")


if __name__ == "__main__":
    unittest.main()
