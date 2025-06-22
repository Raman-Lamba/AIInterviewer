import os
import json
import argparse
from groq import Groq
from real_time import extract_text_from_pdf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_json_file(path):
    """Loads a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_interview(candidate_id, interview_name):
    """
    Evaluates a candidate's interview performance by sending their answers,
    the ideal answers, and their resume to the Groq API.
    """
    try:
        # 1. Load all necessary files
        print("Loading files...")
        resume_path = os.path.join('resumes', f'r{candidate_id}.pdf')
        transcript_path = os.path.join('interview_transcript', f'candidate_{candidate_id}_{interview_name}.json')
        questions_path = os.path.join('questions', f'{interview_name}.json')

        resume_text = extract_text_from_pdf(resume_path)
        transcript = load_json_file(transcript_path)
        questions_data = load_json_file(questions_path)
        
        # Create a mapping of question_id to ideal answer for easier lookup
        key = list(questions_data.keys())[0]
        ideal_answers = {q['id']: q['answer'] for q in questions_data[key]['questions']}
        
        # 2. Prepare the data for the prompt
        evaluation_data = []
        for item in transcript:
            question_id = item['question_id']
            evaluation_data.append({
                "question": item['question'],
                "candidate_answer": item['answer'],
                "ideal_answer_points": ideal_answers.get(question_id, "N/A (evaluate based on general knowledge and resume context)")
            })

        # 3. Create the prompt for Groq
        print("Generating evaluation prompt...")
        prompt = f"""
        As an expert technical interviewer, please evaluate the following interview answers.
        The candidate's resume is provided for context.
        For each question, provide a score from 1 to 10 and a concise reason for the score.
        The ideal answer key points are provided for pre-recorded questions. For resume-based questions, evaluate based on the candidate's resume and the clarity of their response.

        Candidate Resume:
        ---
        {resume_text}
        ---

        Interview Questions and Answers:
        ---
        {json.dumps(evaluation_data, indent=2)}
        ---

        Please return your evaluation as a JSON object with a single key "evaluation", which is a list of objects.
        Each object should have three keys: "question", "score", and "reason".
        Example: {{"evaluation": [{{"question": "...", "score": 8, "reason": "..."}}]}}
        """

        # 4. Call the Groq API
        print("Sending request to Groq for evaluation...")
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert hiring manager providing interview feedback in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="deepseek-r1-distill-llama-70b",
            response_format={"type": "json_object"},
        )
        
        evaluation_result = json.loads(chat_completion.choices[0].message.content)
        return evaluation_result

    except Exception as e:
        print(f"❌ An error occurred during evaluation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate a candidate's interview.")
    parser.add_argument('--id', dest='candidate_id', type=int, required=True, help="Candidate ID")
    parser.add_argument('--interview', dest='interview_name', type=str, default='ml', help="Name of the interview (e.g., ml, frontend)")
    args = parser.parse_args()

    result = evaluate_interview(candidate_id=args.candidate_id, interview_name=args.interview_name)
    
    if result:
        print("\n--- Evaluation Result ---")
        for evaluation in result.get("evaluation", []):
            print(f"Question: {evaluation['question']}")
            print(f"  Score: {evaluation['score']}/10")
            print(f"  Reason: {evaluation['reason']}\n")
        
        # Optionally, save the result to a file
        eval_dir = 'evaluations'
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        filename = os.path.join(eval_dir, f'evaluation_{args.candidate_id}_{args.interview_name}.json')
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"✅ Evaluation saved to {filename}")

if __name__ == "__main__":
    main()
