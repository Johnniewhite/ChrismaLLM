import re

def preprocess_text(text):
    # Convert to lowercase, remove special characters and newlines
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def prepare_data(raw_data, output_file):
    # Join all sections of the resume into a single string
    raw_text = " ".join(str(value) for value in raw_data.values())
    processed_text = preprocess_text(raw_text)
    
    with open(output_file, 'w') as f:
        f.write(processed_text)

if __name__ == '__main__':
    # This is just for testing the module directly
    sample_data = {
        'contact': 'John Doe, johndoe@email.com',
        'experience': 'Software Engineer, 5 years',
        'education': 'BS in Computer Science'
    }
    prepare_data(sample_data, 'data/processed/processed_data.txt')