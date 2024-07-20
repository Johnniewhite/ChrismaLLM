import torch

def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token = outputs[0, -1, :].argmax()
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if next_token == tokenizer.tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids.squeeze().tolist())