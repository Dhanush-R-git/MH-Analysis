``` bash
from fastapi import FastAPI, Request # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from pydantic import BaseModel # type: ignore
import logging
import os
from huggingface_hub import InferenceClient # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
if not HF_INFERENCE_API_KEY:
    raise ValueError("HF_INFERENCE_API_KEY missing. Please set it.")

# Initialize InferenceClient for MentaNow questions
mentanow_client = InferenceClient(
    provider="hyperbolic",
    api_key=HF_INFERENCE_API_KEY
)

app = FastAPI()

# Simple question flow (example)
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

class MentaNowRequest(BaseModel):
    userAnswer: str = None
    userAnswers: list = None  # array of {question, answer} objects

@app.post("/api/mentanow")
def mentanow_api(payload: MentaNowRequest):
    """
    This endpoint returns either the next question or a final 'report'.
    In a real scenario, you'd track the user's state in a database or session.
    For simplicity, we track question index by length of userAnswers.
    """
    user_answers = payload.userAnswers or []
    current_index = len(user_answers)  # number of answered questions

    # If not done, return next question
    if current_index < len(QUESTION_FLOW):
        next_question = QUESTION_FLOW[current_index]
        return JSONResponse(content=next_question)
    
    # Otherwise, generate final 'report' using DeepSeek-R1 or some logic
    # We'll build a prompt from userAnswers and call the model
    # Example: Summarize user's concerns
    prompt = build_mentanow_prompt(user_answers)
    logger.info(f"MentaNow final prompt:\n{prompt}")

    # Call DeepSeek-R1 for final summary
    messages = [{"role": "user", "content": prompt}]
    try:
        response_text = ""
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

    return JSONResponse(content={
        "type": "report",
        "reportText": final_report
    })

def build_mentanow_prompt(user_answers):
    """
    Build a simple prompt summarizing the user's answers
    so the model can produce a final mental health 'report' or summary.
    """
    summary_lines = []
    for i, ans in enumerate(user_answers, 1):
        summary_lines.append(f"Q{i}: {ans['question']}\nA{i}: {ans['answer']}\n")

    # Combine into a single string
    summary_text = "\n".join(summary_lines)
    prompt = f"""
You are a mental health assessment model. 
The user answered these questions:
{summary_text}

Provide a concise summary of the user's mental state and suggestions.
Remember: This is not a medical diagnosis, just supportive guidance.
    """
    return prompt.strip()
```