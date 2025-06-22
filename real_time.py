import fitz  # PyMuPDF
from groq import Groq
import os

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume PDF not found at {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def generate_resume_questions(resume_text):
    """
    Uses Groq to generate three insightful questions based on the resume text.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"""
    Based on the following resume, please generate exactly three insightful and concise interview questions.
    The questions should be suitable for a verbal interview.
    Return the questions as a JSON list of strings. For example: ["Question 1?", "Question 2?", "Question 3?"]

    Resume Text:
    ---
    {resume_text}
    ---
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert technical recruiter."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        # The response is a JSON string, so we need to parse it.
        # Assuming the response is like `{"questions": [...]}`
        import json
        questions = json.loads(response_content)
        return questions.get("questions", [])

    except Exception as e:
        print(f"❌ Error getting questions from Groq: {e}")
        return []
