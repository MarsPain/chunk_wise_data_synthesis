import unittest

from path_setup import ensure_src_path

ensure_src_path()

from chunking import (
    detect_dominant_script,
    get_adaptive_length,
    split_into_structural_units,
    split_into_lines,
    split_into_char_chunks,
    split_document_into_chunks,
)
from pipelines import stitch_rewritten_chunks
from tokenization import WhitespaceTokenizer


class ScriptDetectionTests(unittest.TestCase):
    def test_detect_latin(self) -> None:
        self.assertEqual(detect_dominant_script("Hello world"), "latin")
        self.assertEqual(detect_dominant_script("The quick brown fox"), "latin")
    
    def test_detect_cjk(self) -> None:
        self.assertEqual(detect_dominant_script("今天天气很好"), "cjk")
        self.assertEqual(detect_dominant_script("こんにちは世界"), "cjk")  # 日文
        self.assertEqual(detect_dominant_script("안녕하세요"), "cjk")  # 韩文
    
    def test_detect_mixed(self) -> None:
        # 中英文混合，CJK 比例和拉丁比例都不够高
        self.assertEqual(detect_dominant_script("Hi 你好"), "mixed")
        # 纯数字和标点
        self.assertEqual(detect_dominant_script("123 456 !!!"), "mixed")


class AdaptiveLengthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()
    
    def test_auto_mode_latin(self) -> None:
        text = "a b c d e"  # 5 tokens
        self.assertEqual(get_adaptive_length(text, self.tokenizer, "auto"), 5)
    
    def test_auto_mode_cjk(self) -> None:
        text = "今天天气很好"  # 5 chars, 1 token
        # 中文应该返回字符数
        self.assertEqual(get_adaptive_length(text, self.tokenizer, "auto"), 6)  # 6 chars
    
    def test_token_mode(self) -> None:
        text = "今天天气很好"
        self.assertEqual(get_adaptive_length(text, self.tokenizer, "token"), 1)
    
    def test_char_mode(self) -> None:
        text = "a b c d e"
        self.assertEqual(get_adaptive_length(text, self.tokenizer, "char"), 9)


class SplitIntoLinesTests(unittest.TestCase):
    def test_basic_lines(self) -> None:
        text = "line1\nline2\nline3"
        self.assertEqual(split_into_lines(text), ["line1", "line2", "line3"])
    
    def test_empty_lines_removed(self) -> None:
        text = "line1\n\nline2\n  \nline3"
        self.assertEqual(split_into_lines(text), ["line1", "line2", "line3"])
    
    def test_single_line(self) -> None:
        self.assertEqual(split_into_lines("only one line"), ["only one line"])


class SplitIntoCharChunksTests(unittest.TestCase):
    def test_basic_split(self) -> None:
        text = "abcdefghij"
        chunks = split_into_char_chunks(text, 3)
        self.assertEqual(chunks, ["abc", "def", "ghi", "j"])
    
    def test_exact_division(self) -> None:
        text = "abcdef"
        chunks = split_into_char_chunks(text, 3)
        self.assertEqual(chunks, ["abc", "def"])
    
    def test_empty_string(self) -> None:
        self.assertEqual(split_into_char_chunks("", 5), [])


class DocumentChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()
    
    def test_paragraph_level_split(self) -> None:
        """测试段落级分割 - 短段落会累积合并"""
        text = "Para one\n\nPara two\n\nPara three"
        chunks = split_document_into_chunks(
            text, self.tokenizer, chunk_size=100, length_mode="char"
        )
        # 总长度小于100，合并为1个chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Para one Para two Para three")
    
    def test_paragraph_no_merge(self) -> None:
        """测试段落不合并（当单个段落就超限时）"""
        text = "Para one\n\nPara two is longer\n\nPara three is also longer"
        chunks = split_document_into_chunks(
            text, self.tokenizer, chunk_size=10, length_mode="char"
        )
        # 每个段落都超过10字符，不会累积，会触发行或字符分割
        self.assertGreater(len(chunks), 1)
    
    def test_line_fallback_for_long_paragraph(self) -> None:
        """测试超长段落触发单行分割 - 行在限制内时直接保留"""
        text = "Line1\nLine2"
        chunks = split_document_into_chunks(
            text, self.tokenizer, chunk_size=10, length_mode="char"
        )
        # 整段超长（11字符），触发按行分割；行分割后两行可以合并到同一个chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Line1 Line2")
    
    def test_char_fallback_for_long_line(self) -> None:
        """测试超长单行触发字符分割"""
        text = "Line one is here\nLine two is here"  # 无空行，单行16字符
        chunks = split_document_into_chunks(
            text, self.tokenizer, chunk_size=10, length_mode="char"
        )
        # 无空行 → 整段超长 → 按行分割 → 行也超长 → 按字符分割
        self.assertGreater(len(chunks), 1)
        # 验证总内容（注意：字符分割会丢失换行）
        self.assertEqual("".join(chunks), "Line one is hereLine two is here")
    
    def test_chinese_content(self) -> None:
        """测试中文内容"""
        text = "第一行\n第二行\n第三行很长很长很长"
        chunks = split_document_into_chunks(
            text, self.tokenizer, chunk_size=6, length_mode="auto"
        )
        # 中文应该用字符数，每行 3-9 字符
        self.assertGreaterEqual(len(chunks), 2)
    
    def test_accumulation_within_limit(self) -> None:
        """测试在限制内的内容会累积"""
        text = "Short\n\nAlso short"
        chunks = split_document_into_chunks(
            text, self.tokenizer, chunk_size=100, length_mode="char"
        )
        # 两个短段落可以合并到一个 chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short Also short")
    
    def test_disable_fallback(self) -> None:
        """测试禁用降级分割"""
        text = "A very long line that exceeds limit"
        chunks = split_document_into_chunks(
            text, self.tokenizer, chunk_size=10, length_mode="char",
            enable_line_fallback=False, enable_char_fallback=False
        )
        # 不分割，直接返回
        self.assertEqual(len(chunks), 1)


class StitchRewrittenChunksTests(unittest.TestCase):
    def test_stitch_deduplicates_boundary_overlap(self) -> None:
        tokenizer = WhitespaceTokenizer()
        merged = stitch_rewritten_chunks(
            chunks=["alpha beta gamma", "gamma delta epsilon"],
            tokenizer=tokenizer,
            max_overlap_tokens=4,
        )
        self.assertEqual(merged, "alpha beta gamma delta epsilon")


if __name__ == "__main__":
    unittest.main()
