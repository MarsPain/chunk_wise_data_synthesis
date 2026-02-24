import unittest
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from prompting import RewriteRequest, render_rewrite_prompt


class RewritePromptLanguageTests(unittest.TestCase):
    def _build_request(self, prompt_language: str = "en") -> RewriteRequest:
        return RewriteRequest(
            style_instruction="保持原意并提升可读性",
            global_anchor="锚点信息",
            generated_prefix="已生成前缀",
            current_chunk="当前分块",
            retry_index=0,
            strict_fidelity=False,
            prompt_language=prompt_language,
        )

    def test_render_rewrite_prompt_defaults_to_english(self) -> None:
        prompt = render_rewrite_prompt(self._build_request())
        self.assertIn("You are a faithful rewriter.", prompt)
        self.assertIn("Current chunk:", prompt)

    def test_render_rewrite_prompt_supports_chinese(self) -> None:
        prompt = render_rewrite_prompt(self._build_request(prompt_language="zh"))
        self.assertIn("你是一名忠实改写助手。", prompt)
        self.assertIn("当前分块：", prompt)


if __name__ == "__main__":
    unittest.main()
