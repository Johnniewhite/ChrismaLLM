import re

def process_resume_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract key information
    sections = {
        'contact': re.findall(r'Contact:(.*?)WORK EXPERIENCE:', content, re.DOTALL),
        'work_experience': re.findall(r'WORK EXPERIENCE:(.*?)EXTRACURRICULAR ACTIVITIES:', content, re.DOTALL),
        'extracurricular': re.findall(r'EXTRACURRICULAR ACTIVITIES:(.*?)EDUCATION:', content, re.DOTALL),
        'education': re.findall(r'EDUCATION:(.*?)SKILLS:', content, re.DOTALL),
        'skills': re.findall(r'SKILLS:(.*?)AI ENGINEERING EXPERIENCE AND PROJECTS:', content, re.DOTALL),
        'ai_projects': re.findall(r'AI ENGINEERING EXPERIENCE AND PROJECTS:(.*?)(?:\Z)', content, re.DOTALL)
    }

    processed_data = {}
    for key, value in sections.items():
        processed_data[key] = value[0].strip() if value else ''

    # Further process skills into categories
    skills = processed_data['skills'].split('\n\n')
    processed_data['skills'] = {skill.split(':')[0].strip(): skill.split(':')[1].strip() for skill in skills if ':' in skill}

    return processed_data