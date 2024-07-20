import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer(embedded, embedded)
        return self.fc(output)