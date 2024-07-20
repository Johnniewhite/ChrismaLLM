import unittest
import torch
from src.model import SimpleTransformer

class TestSimpleTransformer(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.d_model = 128
        self.nhead = 4
        self.num_layers = 2
        self.model = SimpleTransformer(self.vocab_size, self.d_model, self.nhead, self.num_layers)

    def test_forward_pass(self):
        batch_size = 2
        seq_length = 10
        input_tensor = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (batch_size, seq_length, self.vocab_size))

if __name__ == '__main__':
    unittest.main()