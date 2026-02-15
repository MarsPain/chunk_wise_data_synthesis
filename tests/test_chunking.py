import unittest

from path_setup import ensure_src_path

ensure_src_path()

from chunking import split_into_token_chunks
from pipeline import stitch_rewritten_chunks
from tokenizer import WhitespaceTokenizer


class ChunkingTests(unittest.TestCase):
    def test_split_into_token_chunks_with_overlap(self) -> None:
        tokenizer = WhitespaceTokenizer()
        text = " ".join(f"t{i}" for i in range(10))

        chunks = split_into_token_chunks(
            text=text,
            tokenizer=tokenizer,
            chunk_tokens=4,
            overlap_tokens=1,
        )

        self.assertEqual(
            chunks,
            [
                "t0 t1 t2 t3",
                "t3 t4 t5 t6",
                "t6 t7 t8 t9",
            ],
        )

    def test_stitch_rewritten_chunks_deduplicates_boundary_overlap(self) -> None:
        tokenizer = WhitespaceTokenizer()
        merged = stitch_rewritten_chunks(
            chunks=["alpha beta gamma", "gamma delta epsilon"],
            tokenizer=tokenizer,
            max_overlap_tokens=4,
        )
        self.assertEqual(merged, "alpha beta gamma delta epsilon")


if __name__ == "__main__":
    unittest.main()
