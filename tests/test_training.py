import unittest
import torch
from src.tokenizer import SimpleTokenizer
from src.model import SimpleTransformer
from src.training import train_model

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleTokenizer()
        self.model = SimpleTransformer(self.tokenizer.get_vocab_size(), d_model=128, nhead=4, num_layers=2)
        
        # Create a larger test dataset
        self.test_data = "This is a small test dataset for training. " * 100
        with open('test_data.txt', 'w') as f:
            f.write(self.test_data)

    def test_train_model(self):
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        trained_model = train_model(self.model, self.tokenizer, 'test_data.txt', batch_size=32, epochs=5)
        
        # Check if model parameters have been updated
        for name, param in trained_model.named_parameters():
            self.assertFalse(torch.allclose(initial_params[name], param, atol=1e-4), 
                             f"Parameter {name} was not significantly updated during training")

    def tearDown(self):
        import os
        os.remove('test_data.txt')

if __name__ == '__main__':
    unittest.main()