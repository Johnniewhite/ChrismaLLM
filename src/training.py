import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, tokenizer, data_file, batch_size=32, epochs=10, lr=0.001):
    with open(data_file, 'r') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    tensor_data = torch.tensor(tokens)
    
    dataset = TensorDataset(tensor_data[:-1], tensor_data[1:])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()  # Set the model to training mode
    
    for epoch in range(epochs):
        total_loss = 0
        for batch, (input_data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch}, Loss: {loss.item()}')
        
        print(f'Epoch {epoch}, Average Loss: {total_loss / len(dataloader)}')
    
    return model