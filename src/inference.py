import torch

def generate_response(model, tokenizer, prompt, resume_data):
    model.eval()
    
    # Prepare context
    context = f"""Resume Data:
Contact: {resume_data['contact']}
Work Experience: {resume_data['work_experience']}
Extracurricular Activities: {resume_data['extracurricular']}
Education: {resume_data['education']}
Skills: {', '.join([f"{k}: {v}" for k, v in resume_data['skills'].items()])}
AI Engineering Experience and Projects: {resume_data['ai_projects']}

Question: {prompt}
Answer:"""
    
    tokens = tokenizer.encode(context)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    
    max_length = 150  # Increased to allow for longer responses
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token = outputs[0, -1, :].argmax()
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if next_token == tokenizer.tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids.squeeze().tolist())
    
    # Extract the answer part
    answer = generated_text.split("Answer:")[-1].strip()
    
    return answer