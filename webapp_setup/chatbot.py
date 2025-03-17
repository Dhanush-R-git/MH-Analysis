from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os
import uvicorn
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import logging
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Hugging Face token for mental state detection model
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token is missing. Set the environment variable HUGGINGFACE_TOKEN.")

# Load Inference API key for our Llama-based instruct model
HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
if not HF_INFERENCE_API_KEY:
    raise ValueError("Inference API key is missing. Set the environment variable HF_INFERENCE_API_KEY.")

# Initialize the InferenceClient
client = InferenceClient(
    provider="nebius",  # Use the HyperBolic provider
    api_key=HF_INFERENCE_API_KEY,
)

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="/workspaces/MHRoberta-a-LLM-for-mental-health-analysis/webapp_setup/templates")

# Load mental health model (gated repo)
try:
    tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base", use_auth_token=HF_TOKEN)
    model = AutoModelForMaskedLM.from_pretrained("mental/mental-roberta-base", use_auth_token=HF_TOKEN)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Pydantic model for chat input validation
class ChatRequest(BaseModel):
    message: str

def detect_mental_state(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_token_id = torch.argmax(logits, dim=-1)
    mental_state = tokenizer.decode(predicted_token_id[0])
    return mental_state

def get_chatbot_response(mental_state: str, user_message: str) -> str:
    """
    Builds a few-shot prompt tailored to the detected mental state and
    uses the Hugging Face InferenceClient to generate a response.
    """
    few_shot_examples = (
        "### Example 1:\n"
        "[Condition: sadness]\n"
        "User: \"I'm feeling so down and hopeless.\"\n"
        "Assistant: \"I'm truly sorry you're feeling this way. Sometimes it helps to talk through your feelings. Consider reaching out to someone you trust or a professional.\"\n\n"
        "### Example 2:\n"
        "[Condition: anxiety]\n"
        "User: \"I feel anxious about everything.\"\n"
        "Assistant: \"It sounds like you're under a lot of stress. Remember, deep breathing exercises and short breaks might help calm your mind.\"\n\n"
        "### Example 3:\n"
        "[Condition: anger]\n"
        "User: \"I feel angry and frustrated all the time.\"\n"
        "Assistant: \"It's okay to feel anger sometimes. Try to channel that energy into something creative or physical, and consider talking to someone about how you feel.\"\n\n"
    )

    prompt = (
        f"{few_shot_examples}\n"
        f"Now, the user is experiencing {mental_state}.\n"
        f"User: \"{user_message}\"\n"
        "Assistant:"
    )
    logger.info(f"Generated prompt: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    try:
        # Use the InferenceClient with streaming; accumulate the response text.
        stream = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=500,
            stream=True,
        )
        response_text = ""
        for chunk in stream:
            # Accumulate text from each streaming chunk.
            response_text += chunk.choices[0].delta.content
        logger.info(f"Inference response: {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"Error in Inference API: {e}")
        return "I'm sorry, I'm having trouble understanding you right now."

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Renders the landing page (or chatbot UI) template
    return templates.TemplateResponse("web.html", {"request": request})

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    # Renders the chatbot UI template
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received request data: {data}")

        user_message = data.get("user_message", "").strip()
        system_prompt = data.get("system_prompt", "").strip()
        model_name = data.get("model_name", "").strip()
        temperature = data.get("temperature", 0.7)

        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Message cannot be empty."})

        mental_state = detect_mental_state(user_message)
        chatbot_response = get_chatbot_response(mental_state, user_message)

        return JSONResponse(content={"mental_state": mental_state, "response": chatbot_response})
    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="0.0.0.0", port=5000, reload=True)
