import os
import re

def load_data(data_dir):
    data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(data_dir, file_name), 'r') as file:
                data.append(file.read())
    return ' '.join(data)


def preprocess_text(text):
    # convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

def prepare_data(data_dir, output_file):
    raw_text = load_data(data_dir)
    processed_text = preprocess_text(raw_text)
    
    with open(output_file, 'w') as f:
        f.write(processed_text)

if __name__ == '__main__':
    prepare_data('data/raw', 'data/processed/processed_data.txt')