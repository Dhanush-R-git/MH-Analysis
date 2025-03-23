import os
import logging
from huggingface_hub import InferenceClient # type: ignore
from dotenv import load_dotenv # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
# Validate and load environment variable for the Inference API key
HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
if not HF_INFERENCE_API_KEY:
    raise ValueError("HF_INFERENCE_API_KEY missing. Please set it.")

# Initialize InferenceClient for MentaNow questions
mentanow_client = InferenceClient(
    provider="hyperbolic",
    api_key=HF_INFERENCE_API_KEY
)

# Define the question flow for MentaNow
QUESTION_FLOW = [
    {
        "title": "Are you answering for yourself or someone else?",
        "text": "Please select one option.",
        "type": "radio",
        "options": ["Myself", "Someone else"]
    },
    {
        "title": "What is your age range?",
        "text": "Choose your age range.",
        "type": "radio",
        "options": ["Under 18", "18-30", "31-50", "50+"]
    },
    {
        "title": "What is your primary concern right now?",
        "text": "Briefly describe what is troubling you (e.g., stress, anxiety).",
        "type": "text"
    }
    # Add more questions as needed
]

def build_mentanow_prompt(user_answers):
    """
    Build a prompt summarizing the user's answers so that the model
    can produce a final mental health 'report' or summary.
    """
    summary_lines = []
    for i, ans in enumerate(user_answers, 1):
        summary_lines.append(f"Q{i}: {ans['question']}\nA{i}: {ans['answer']}\n")
    summary_text = "\n".join(summary_lines)
    prompt = f"""
You are a mental health assessment model. 
The user answered these questions:
{summary_text}

Provide a concise summary of the user's mental state and suggestions.
Remember: This is not a medical diagnosis, just supportive guidance.
    """
    return prompt.strip()

def generate_mentanow_report(user_answers):
    """
    Generate a final report using the user's answers.
    The report is created by building a prompt and then calling the
    inference client with a streaming request.
    """
    prompt = build_mentanow_prompt(user_answers)
    logger.info(f"MentaNow final prompt:\n{prompt}")
    
    messages = [{"role": "user", "content": prompt}]
    response_text = ""
    try:
        stream = mentanow_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages,
            max_tokens=300,
            stream=True
        )
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        final_report = response_text.strip()
    except Exception as e:
        logger.error(f"Error calling DeepSeek-R1: {e}")
        final_report = "Sorry, something went wrong while generating your report."
    
    return final_report
