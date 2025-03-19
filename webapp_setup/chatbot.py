from fastapi import FastAPI, Request, HTTPException, Response # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse # type: ignore
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torch # type: ignore
import os
import uvicorn # type: ignore
from pydantic import BaseModel  # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore
import logging
from huggingface_hub import InferenceClient # type: ignore
import asyncio

# Configuration (externalized via environment variables)
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
MENTAL_MODEL_NAME = os.getenv("MENTAL_MODEL_NAME", "mental/mental-roberta-base")
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", 100))
INFERENCE_MAX_NEW_TOKENS = int(os.getenv("INFERENCE_MAX_NEW_TOKENS", 200))
LOCAL_TOP_K = int(os.getenv("LOCAL_TOP_K", 50))
LOCAL_TOP_P = float(os.getenv("LOCAL_TOP_P", 0.95))
LOCAL_NO_REPEAT_NGRAM_SIZE = int(os.getenv("LOCAL_NO_REPEAT_NGRAM_SIZE", 2))
LOCAL_TRUNCATION_LENGTH = int(os.getenv("LOCAL_TRUNCATION_LENGTH", 400))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables for Hugging Face tokens
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token is missing. Set the environment variable HUGGINGFACE_TOKEN.")

HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
if not HF_INFERENCE_API_KEY:
    raise ValueError("Inference API key is missing. Set the environment variable HF_INFERENCE_API_KEY.")

# Initialize the InferenceClient for cloud inference
client = InferenceClient(api_key=HF_INFERENCE_API_KEY)

# Initialize FastAPI app and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="D:/project-main/Final-year-projects/Project-Laboratory/MH-Analysis/webapp_setup/templates")

# Global variables for lazy-loading the local model
local_model = None
local_tokenizer = None

def load_local_model():
    """Lazy loads the local model and tokenizer if not already loaded."""
    global local_model, local_tokenizer
    if local_model is None or local_tokenizer is None:
        try:
            logger.info("Lazy loading local model...")
            local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, token=HF_TOKEN)
            local_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, token=HF_TOKEN)
            logger.info("Local fallback model loaded successfully with offloading.")
        except Exception as e:
            logger.exception("Failed to load local fallback model")
            raise RuntimeError("Local model not available")
    return local_model, local_tokenizer

# Load mental health model at startup (always loaded)
try:
    logger.info("Loading mental health model...")
    mental_tokenizer = AutoTokenizer.from_pretrained(MENTAL_MODEL_NAME, token=HF_TOKEN)
    mental_model = AutoModelForMaskedLM.from_pretrained(MENTAL_MODEL_NAME, token=HF_TOKEN)
    logger.info("Mental health model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load mental health model")
    raise RuntimeError(f"Failed to load mental health model: {e}")

class ChatRequest(BaseModel):
    user_message: str
    system_prompt: str
    model_name: str
    temperature: float

def detect_mental_state(user_message: str) -> str:
    """Detect mental state using the mental health model."""
    inputs = mental_tokenizer(user_message, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = mental_model(**inputs)
    # Process logits to derive a mental state (customize as needed)
    logits = outputs.logits[:, :-1, :].mean(dim=1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probs, dim=-1)
    mental_state = mental_tokenizer.decode(predicted_token_id)
    return mental_state

def get_chatbot_response(mental_state: str, user_message: str, 
                        system_prompt: str, model_name: str, 
                        temperature: float) -> str:
    """Generate a chatbot response based on mental state and user input."""
    system_instruction = """
You are a compassionate and supportive mental health chatbot. Your goal is to provide empathetic, actionable advice to help users manage their emotional challenges.
"""
    few_shot_examples = """
### Example 0:
User: "Hi or Hello"
Assistant: "Hello! How can I support you today?"

### Example 1:
[Condition: sadness]
User: "I feel angry and frustrated all the time."
Assistant: "It's completely normal to feel frustrated from time to time. Have you tried talking to someone about it? Sometimes sharing what's on your mind can help you process your emotions and gain a new perspective."

**INSTRUCTIONS:**
- Recognize the user's emotion explicitly (e.g., "It's normal to feel frustrated").
- Offer a supportive suggestion (e.g., "Have you tried talking to someone?").
- Encourage an action that can help process emotions (e.g., "Sharing your thoughts can help").

### Example 2:
[Condition: anxiety]
User: "I feel anxious about everything."
Assistant: "It sounds like you're feeling overwhelmed. Try focusing on your breathing for a few minutes. Deep breathing exercises can help calm your mind and bring you back to the present."

**INSTRUCTIONS:**
- Acknowledge the user's feelings and validate them.
- Suggest a practical coping strategy (e.g., "Try focusing on your breathing").
- Offer reassurance that the feeling is temporary and manageable.

### Example 3:
[Condition: sadness]
User: "I feel really low and unmotivated."
Assistant: "I'm sorry you're feeling this way. You might find it helpful to engage in small, enjoyable activities, even if they feel hard to start. A short walk, listening to music, or journaling your thoughts can be helpful."

**INSTRUCTIONS:**
- Express empathy and understanding.
- Suggest small, achievable actions the user can take.
- Avoid forcing positivity; instead, gently encourage self-care.
"""
    conversation_prompt = f"""
{system_instruction}
{system_prompt}

{few_shot_examples}
**New Conversation:

[Condition: {mental_state}]
User: "{user_message}"
Assistant:

"""
    generated = ""
    error = None
    try:
        if model_name == "Inference-Provider":
            response_text = client.text_generation(
                model=LOCAL_MODEL_NAME,
                prompt=conversation_prompt,
                max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
                temperature=temperature
            )
            generated = response_text.strip()
            logger.info(f"Inference Provider Response:\n{generated}")
        elif model_name == "Local-Provider":
            try:
                local_model_loaded, local_tokenizer_loaded = load_local_model()
            except Exception as e:
                logger.exception("Local model error")
                return "I'm sorry, I'm having trouble understanding you right now."
                
            if local_tokenizer_loaded.pad_token is None:
                local_tokenizer_loaded.pad_token = local_tokenizer_loaded.eos_token
            local_inputs = local_tokenizer_loaded(
                conversation_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=LOCAL_TRUNCATION_LENGTH
            )
            local_outputs = local_model_loaded.generate(
                input_ids=local_inputs.input_ids,
                attention_mask=local_inputs.attention_mask,
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature,
                top_k=LOCAL_TOP_K,
                top_p=LOCAL_TOP_P,
                no_repeat_ngram_size=LOCAL_NO_REPEAT_NGRAM_SIZE,
                pad_token_id=local_tokenizer_loaded.eos_token_id
            )
            generated = local_tokenizer_loaded.decode(local_outputs[0], skip_special_tokens=True).strip()
            logger.info(f"Local Model Response:\n{generated}")
        else:
            raise ValueError(f"Unknown model selected: {model_name}")
    except Exception as e:
        logger.exception(f"Error in {model_name} inference")
        error = f"Error using {model_name}: {str(e)}"
        return "I'm sorry, I'm having trouble understanding you right now."
    if error:
        return f"⚠️ {error} - I'm sorry, I'm having trouble understanding you right now."
    assistant_reply = generated.strip()
    if "Assistant:" in assistant_reply:
        assistant_reply = assistant_reply.split("Assistant:", 1)[-1].strip()
    if "### New Conversation:" in assistant_reply:
        assistant_reply = assistant_reply.split("### New Conversation:")[0].strip()
    logger.info(f"Final Extracted Assistant Reply:\n{assistant_reply}")
    return assistant_reply

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

@app.get("/Overview", response_class=HTMLResponse)
async def overview_page(request: Request):
    return templates.TemplateResponse("Overview.html", {"request": request})

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/api/model-status")
async def model_status():
    # Check if the local model can be loaded
    local_status = False
    try:
        _ = load_local_model()
        local_status = True
    except Exception:
        local_status = False
    return {
        "Local-Provider": local_status,
        "Inference-Provider": HF_INFERENCE_API_KEY is not None
    }

@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    """Chat API endpoint."""
    try:
        user_message = chat_request.user_message.strip()
        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Message cannot be empty."})
        
        # Offload blocking calls to background threads
        mental_state = await asyncio.to_thread(detect_mental_state, user_message)
        chatbot_response = await asyncio.to_thread(
            get_chatbot_response,
            mental_state,
            user_message,
            chat_request.system_prompt,
            chat_request.model_name,
            chat_request.temperature
        )
        return JSONResponse(content={"mental_state": mental_state, "response": chatbot_response})
    except Exception as e:
        logger.exception("Error in /api/chat endpoint")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="127.0.0.1", port=8000, reload=True)
