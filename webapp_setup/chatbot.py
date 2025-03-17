from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
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

# Initialize the InferenceClient (client-based inference)
client = InferenceClient(
    provider="nebius",  # Use the HyperBolic provider
    api_key=HF_INFERENCE_API_KEY,
)

# Attempt to load the local fallback model.
try:
    logger.info("Attempting to load local fallback model...")
    local_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_auth_token=HF_TOKEN)
    local_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_auth_token=HF_TOKEN)
    logger.info("Local fallback model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load local fallback model: {e}")
    local_tokenizer = None
    local_model = None

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates (ensure this path is correct in your environment)
templates = Jinja2Templates(directory="/workspaces/MHRoberta-a-LLM-for-mental-health-analysis/webapp_setup/templates")

# Load mental health model (gated repo)
try:
    logger.info("Attempting to load mental health model...")
    tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base", use_auth_token=HF_TOKEN)
    model = AutoModelForMaskedLM.from_pretrained("mental/mental-roberta-base", use_auth_token=HF_TOKEN)
    logger.info("Mental health model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load mental health model: {e}")

# Pydantic model for validating incoming chat requests from the frontend.
class ChatRequest(BaseModel):
    user_message: str
    system_prompt: str
    model_name: str
    temperature: float

def detect_mental_state(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_token_id = torch.argmax(logits, dim=-1)
    mental_state = tokenizer.decode(predicted_token_id[0])
    return mental_state

def get_chatbot_response(mental_state: str, user_message: str, system_prompt: str, model_choice: str, temperature: float) -> str:
    """
    Generates a chatbot response based on the user's message and detected mental state.
    Ensures the model does not repeat the entire prompt in its response.
    """
    
    few_shot_examples = """
    ### Example 1:
    [Condition: sadness]
    User: "I'm feeling so down and hopeless."
    Assistant: "I'm truly sorry you're feeling this way. Sometimes it helps to talk through your feelings. Consider reaching out to someone you trust or a professional."

    ### Example 2:
    [Condition: anxiety]
    User: "I feel anxious about everything."
    Assistant: "It sounds like you're under a lot of stress. Remember, deep breathing exercises and short breaks might help calm your mind."

    ### Example 3:
    [Condition: anger]
    User: "I feel angry and frustrated all the time."
    Assistant: "It's okay to feel anger sometimes. Try to channel that energy into something creative or physical, and consider talking to someone about how you feel."
    """
    # Properly format the new conversation context
    conversation_prompt = f"""
    {system_prompt}

    {few_shot_examples}

    ### New Conversation:
    [Condition: {mental_state}]
    User: "{user_message}"
    Assistant:
    """
    
    # Combine the system prompt with few-shot examples and conversation context.
    logger.info(f"Generated Prompt:\n{conversation_prompt}")

    messages = [{"role": "user", "content": conversation_prompt}]

        # If the user chooses the inference provider model
    if model_choice == "inference_provider":
        try:
            response_text = ""
            stream = client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                max_tokens=200,  # Adjust for response length
                temperature=temperature,
                stop=["User:", "Assistant:", "\n\n"],  # Stop the model from generating more than needed
                stream=True
            )
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            logger.info(f"Client Model Response: {response_text.strip()}")
            return response_text.strip()
        except Exception as e:
            logger.error(f"Error in inference provider: {e}, switching to local model.")

    # Fallback to local model if the chosen model is local or inference fails
    if local_tokenizer is None or local_model is None:
        logger.error("Local fallback model is not available.")
        return "I'm sorry, I'm having trouble understanding you right now."

    try:
        local_inputs = local_tokenizer(conversation_prompt, return_tensors="pt")
        local_outputs = local_model.generate(
            local_inputs.input_ids,
            max_length=400,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=local_tokenizer.eos_token_id
        )
        local_response_text = local_tokenizer.decode(local_outputs[0], skip_special_tokens=True)

        # Extract only the assistant’s reply
        assistant_reply = local_response_text.split("Assistant:", 1)[-1].strip()
        logger.info(f"Local Model Response: {assistant_reply}")
        return assistant_reply
    except Exception as local_e:
        logger.error(f"Error in local inference: {local_e}")
        return "I'm sorry, I'm having trouble understanding you right now."

# Define the root endpoint

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

# Serve a minimal response for favicon.ico to prevent 404 errors.
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/api/chat")
async def chat(chat_request: ChatRequest, request: Request):
    try:
        user_message = chat_request.user_message.strip()
        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Message cannot be empty."})
        
        mental_state = detect_mental_state(user_message)
        chatbot_response = get_chatbot_response(
            mental_state,
            user_message,
            chat_request.system_prompt,
            chat_request.model_name,
            chat_request.temperature
        )
        return JSONResponse(content={"mental_state": mental_state, "response": chatbot_response})
    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)
