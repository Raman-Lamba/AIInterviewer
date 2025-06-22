from vosk import Model, KaldiRecognizer
import pyaudio
import json
from pynput import keyboard as pynput_keyboard
import threading
import time
from groq import Groq
import os
from dotenv import load_dotenv
import logging
from rime import rime_tts
import tempfile
import sys
import subprocess
import argparse
from real_time import extract_text_from_pdf, generate_resume_questions

# Set up logging for error handling
logging.basicConfig(filename='voice_ai_chat.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s:%(message)s')

# Load environment variables from .env file
load_dotenv()

class AudioHandler:
    def __init__(self, model_path="model"):
        if not os.path.exists(model_path):
            print(f"❌ Vosk model not found at '{model_path}'. Please download a model from https://alphacephei.com/vosk/models and extract it to '{model_path}'")
            exit(1)
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, 16000)
        self.recording = False
        self._stop_recording = False
        self._quit_flag = False
        self.p = pyaudio.PyAudio()

    def listen_for_audio(self, max_duration=30):
        print("\n🎤 Listening... Speak, then press 'R' to stop recording, 'Q' to quit")
        self.recording = True
        self._stop_recording = False
        self._quit_flag = False
        frames = []
        stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        start_time = time.time()

        def on_press(key):
            try:
                if key.char and key.char.lower() == 'r':
                    self._stop_recording = True
                elif key.char and key.char.lower() == 'q':
                    self._quit_flag = True
            except AttributeError:
                pass

        with pynput_keyboard.Listener(on_press=on_press) as listener:
            while self.recording:
                if self._stop_recording or self._quit_flag:
                    self.recording = False
                    if self._stop_recording:
                        print("🛑 Recording stopped!")
                    break
                if time.time() - start_time > max_duration:
                    print("⏰ Maximum recording duration reached.")
                    self.recording = False
                    break
                data = stream.read(4000, exception_on_overflow=False)
                frames.append(data)
                time.sleep(0.1)
            listener.stop()
        stream.stop_stream()
        stream.close()
        if self._quit_flag:
            raise KeyboardInterrupt
        return b"".join(frames)

class AIInterviewer:
    def __init__(self, candidate_id, interview_name):
        self.candidate_id = candidate_id
        self.interview_name = interview_name
        self.transcript = []
        self.questions = self.load_questions()
        self.audio_handler = AudioHandler()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_questions(self):
        """Load questions from the specified JSON file."""
        questions_path = os.path.join('questions', f'{self.interview_name}.json')
        if not os.path.exists(questions_path):
            print(f"❌ Questions file not found at '{questions_path}'")
            exit(1)
        with open(questions_path, 'r') as f:
            data = json.load(f)
            # Assuming the structure is { "topic": { "questions": [...] } }
            key = list(data.keys())[0]
            return data[key]['questions']

    def speak(self, text):
        print(f"AI: {text}")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
                rime_tts(text, tmpfile.name)
                tmpfile.flush()
                try:
                    subprocess.run(['mpg123', tmpfile.name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    subprocess.run(['ffplay', '-nodisp', '-autoexit', tmpfile.name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"❌ Error in TTS: {e}")
            logging.error(f"TTS error: {e}")
    
    def speech_to_text(self, audio_bytes):
        if not audio_bytes:
            return None
        try:
            print("🔄 Converting speech to text...")
            rec = self.audio_handler.rec
            rec.AcceptWaveform(audio_bytes)
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                print(f"You said: {text}")
                return text
            else:
                print("❌ Could not understand audio")
                return None
        except Exception as e:
            print(f"❌ Error during speech-to-text: {e}")
            logging.error(f"Speech-to-text error: {e}")
            return None
    
    def wait_for_key(self):
        """Wait for 'e' to start listening or 'q' to quit."""
        print("\nPress 'E' to start answering, or 'Q' to quit the interview.")
        key_pressed = {'e': False, 'q': False}
        def on_press(key):
            try:
                if key.char and key.char.lower() == 'e':
                    key_pressed['e'] = True
                    return False
                elif key.char and key.char.lower() == 'q':
                    key_pressed['q'] = True
                    return False
            except AttributeError:
                pass
        with pynput_keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        return key_pressed

    def save_transcript(self):
        """Save the interview transcript to a JSON file."""
        transcript_dir = 'interview_transcript'
        if not os.path.exists(transcript_dir):
            os.makedirs(transcript_dir)
        filename = os.path.join(transcript_dir, f'candidate_{self.candidate_id}_{self.interview_name}.json')
        with open(filename, 'w') as f:
            json.dump(self.transcript, f, indent=4)
        print(f"\n✅ Interview complete! Transcript saved to {filename}")

    def _ask_question_and_record(self, question, question_id="dynamic"):
        self.speak(question)
        key = self.wait_for_key()
        if key['q']:
            return "quit"
        if key['e']:
            answer_bytes = self.audio_handler.listen_for_audio()
            if answer_bytes:
                answer_text = self.speech_to_text(answer_bytes)
                self.transcript.append({
                    "question_id": question_id,
                    "question": question,
                    "answer": answer_text or "No answer recorded."
                })
            else:
                self.transcript.append({
                    "question_id": question_id,
                    "question": question,
                    "answer": "No answer recorded."
                })

    def start_interview(self):
        print("🤖 AI Interviewer Started!")
        print("=" * 50)
        
        # Check for API keys
        if not os.getenv("GROQ_API_KEY") or not os.getenv("RIME_API_KEY"):
            print("❌ Please ensure GROQ_API_KEY and RIME_API_KEY are set.")
            return

        # Introduction
        intro = f"Hello, and welcome to your {self.interview_name} interview. My name is Rime, and I'll be guiding you through some questions today. Let's begin."
        self.speak(intro)
        
        try:
            # Phase 1: Pre-recorded questions
            for item in self.questions:
                if self._ask_question_and_record(item['question'], item['id']) == "quit":
                    raise KeyboardInterrupt

            # Phase 2: Resume-based questions
            self.speak("Great. Now I will ask a few questions based on your resume.")
            resume_path = os.path.join('resumes', f'r{self.candidate_id}.pdf')
            try:
                resume_text = extract_text_from_pdf(resume_path)
                resume_questions = generate_resume_questions(resume_text)
                
                if not resume_questions:
                    self.speak("I couldn't generate any questions from your resume, but we'll proceed.")
                else:
                    for question in resume_questions:
                         if self._ask_question_and_record(question) == "quit":
                            raise KeyboardInterrupt

            except FileNotFoundError:
                self.speak("I couldn't find a resume for you, so we'll skip the resume-based questions.")
            except Exception as e:
                self.speak("I ran into an issue processing your resume, so we'll skip that part.")
                print(f"Error during resume phase: {e}")

        except KeyboardInterrupt:
            print("\n👋 Ending interview early.")
        finally:
            self.save_transcript()
            self.speak("Thank you for your time. The interview is now complete.")

def main():
    parser = argparse.ArgumentParser(description="AI Interviewer")
    parser.add_argument('--id', dest='candidate_id', type=int, default=1, help="Candidate ID")
    parser.add_argument('--interview', dest='interview_name', type=str, default='ml', help="Name of the interview (e.g., ml, frontend)")
    args = parser.parse_args()

    print("Setting up AI Interviewer...")
    interviewer = AIInterviewer(candidate_id=args.candidate_id, interview_name=args.interview_name)
    interviewer.start_interview()

if __name__ == "__main__":
    main()