import unittest
from src.tokenizer import SimpleTokenizer

class TestSimpleTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleTokenizer()

    def test_encode_decode(self):
        text = "Hello, world!"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_vocab_size(self):
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertGreater(vocab_size, 0)

if __name__ == '__main__':
    unittest.main()