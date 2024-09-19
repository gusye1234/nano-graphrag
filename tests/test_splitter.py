import unittest
from typing import List
import tiktoken
from nano_graphrag._splitter import SeparatorSplitter
from nano_graphrag._op import chunking_by_seperators

# Assuming the SeparatorSplitter class is already imported


class TestSeparatorSplitter(unittest.TestCase):

    def setUp(self):
        self.tokenize = lambda text: [
            ord(c) for c in text
        ]  # Simple tokenizer for testing
        self.detokenize = lambda tokens: "".join(chr(t) for t in tokens)

    def test_split_with_custom_separator(self):
        splitter = SeparatorSplitter(
            separators=[self.tokenize("\n"), self.tokenize(".")],
            chunk_size=19,
            chunk_overlap=0,
            keep_separator="end",
        )
        text = "This is a test.\nAnother test."
        tokens = self.tokenize(text)
        expected = [
            self.tokenize("This is a test.\n"),
            self.tokenize("Another test."),
        ]
        result = splitter.split_tokens(tokens)

        self.assertEqual(result, expected)

    def test_chunk_size_limit(self):
        splitter = SeparatorSplitter(
            chunk_size=5, chunk_overlap=0, separators=[self.tokenize("\n")]
        )
        text = "1234567890"
        tokens = self.tokenize(text)
        expected = [self.tokenize("12345"), self.tokenize("67890")]
        result = splitter.split_tokens(tokens)
        self.assertEqual(result, expected)

    def test_chunk_overlap(self):
        splitter = SeparatorSplitter(
            chunk_size=5, chunk_overlap=2, separators=[self.tokenize("\n")]
        )
        text = "1234567890"
        tokens = self.tokenize(text)
        expected = [
            self.tokenize("12345"),
            self.tokenize("45678"),
            self.tokenize("7890"),
        ]
        result = splitter.split_tokens(tokens)
        self.assertEqual(result, expected)

    def test_chunking_by_seperators(self):
        encoder = tiktoken.encoding_for_model("gpt-4o")
        text = "This is a test.\nAnother test."
        tokens_list = [encoder.encode(text)]
        doc_keys = ["doc1"]
        results = chunking_by_seperators(tokens_list, doc_keys, encoder)
        assert len(results) == 1
        assert results[0]["chunk_order_index"] == 0
        assert results[0]["full_doc_id"] == "doc1"
        assert results[0]["content"] == text


if __name__ == "__main__":
    unittest.main()
