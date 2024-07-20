import unittest
import os
from src.data_preparation import preprocess_text, prepare_data

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_preprocess_text(self):
        text = "This is a Test! With some Punctuation."
        processed = preprocess_text(text)
        self.assertEqual(processed, "this is a test with some punctuation")

    def test_prepare_data(self):
        test_data = {
            'contact': 'John Doe, johndoe@email.com',
            'experience': 'Software Engineer, 5 years',
            'education': 'BS in Computer Science'
        }
        output_file = os.path.join(self.test_dir, 'processed.txt')
        prepare_data(test_data, output_file)
        
        with open(output_file, 'r') as f:
            processed = f.read()
        
        expected = "john doe johndoe@email.com software engineer 5 years bs in computer science"
        self.assertEqual(processed, expected)

if __name__ == '__main__':
    unittest.main()
