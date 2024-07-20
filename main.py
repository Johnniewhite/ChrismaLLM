import unittest
import sys
import os

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from src.data_preparation import prepare_data
from src.tokenizer import SimpleTokenizer
from src.model import SimpleTransformer
from src.training import train_model
from src.inference import generate_response
from src.data_processor import process_resume_data

def run_tests():
    # Dynamically discover and load tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.join(project_root, 'tests'))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return True if all tests passed, False otherwise
    return result.wasSuccessful()

def main():
    # Run tests first
    print("Running tests...")
    tests_passed = run_tests()

    if not tests_passed:
        print("Tests failed. Exiting program.")
        sys.exit(1)

    print("All tests passed. Proceeding with main program execution.")

    # Prepare data
    resume_file_path = os.path.join(project_root, 'data', 'raw', 'resume.txt')
    raw_data = process_resume_data(resume_file_path)
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.txt')
    prepare_data(raw_data, processed_data_path)
    
    # Initialize tokenizer and model
    tokenizer = SimpleTokenizer()
    model = SimpleTransformer(vocab_size=tokenizer.get_vocab_size(), d_model=256, nhead=4, num_layers=4)
    
    # Train the model
    trained_model = train_model(model, tokenizer, processed_data_path)
    
    # Interactive prompt loop
    while True:
        prompt = input("Enter your question (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        response = generate_response(trained_model, tokenizer, prompt, raw_data)
        print(f"Response: {response}")

if __name__ == '__main__':
    main()