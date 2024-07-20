from src.data_preparation import prepare_data
from src.tokenizer import SimpleTokenizer
from src.model import SimpleTransformer
from src.training import train_model
from src.inference import generate_text

def main():
    # Prepare data
    prepare_data('data/raw', 'data/processed/processed_data.txt')
    
    # Initialize tokenizer and model
    tokenizer = SimpleTokenizer()
    model = SimpleTransformer(vocab_size=tokenizer.get_vocab_size(), d_model=256, nhead=4, num_layers=4)
    
    # Train the model
    trained_model = train_model(model, tokenizer, 'data/processed/processed_data.txt')
    
    # Generate text
    prompt = "My name is"
    generated_text = generate_text(trained_model, tokenizer, prompt)
    print(f"Generated text: {generated_text}")

if __name__ == '__main__':
    main()