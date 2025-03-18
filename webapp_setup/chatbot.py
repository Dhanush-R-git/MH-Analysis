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

# Load environment variables
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token is missing. Set the environment variable HUGGINGFACE_TOKEN.")

HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
if not HF_INFERENCE_API_KEY:
    raise ValueError("Inference API key is missing. Set the environment variable HF_INFERENCE_API_KEY.")

# Initialize the InferenceClient
client = InferenceClient(api_key=HF_INFERENCE_API_KEY)

# Attempt to load local fallback model
try:
    logger.info("Attempting to load local fallback model...")
    local_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
    local_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
    logger.info("Local fallback model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load local fallback model: {e}")
    local_tokenizer, local_model = None, None

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="/workspaces/MHRoberta-a-LLM-for-mental-health-analysis/webapp_setup/templates")

# Load mental health model
try:
    logger.info("Attempting to load mental health model...")
    tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base", token=HF_TOKEN)
    model = AutoModelForMaskedLM.from_pretrained("mental/mental-roberta-base", token=HF_TOKEN)
    logger.info("Mental health model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load mental health model: {e}")

# Pydantic model for validating incoming chat requests
class ChatRequest(BaseModel):
    user_message: str
    system_prompt: str
    model_name: str
    temperature: float

def detect_mental_state(text: str) -> str:
    """Detect mental state using the mental health model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[:, :-1, :].mean(dim=1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probs, dim=-1)

    mental_state = tokenizer.decode(predicted_token_id)
    return mental_state

def get_chatbot_response(mental_state: str, user_message: str, system_prompt: str, model_choice: str, temperature: float) -> str:
    """Generate a chatbot response based on mental state and user input."""
    system_instruction = """
You are a compassionate and supportive mental health chatbot. Your goal is to provide empathetic, actionable advice to help users manage their emotional challenges.
"""
    
    few_shot_examples = """
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
### New Conversation:

[Condition: {mental_state}]
User: "{user_message}"
Assistant:
    
"""

    messages = [{"role": "user", "content": conversation_prompt}]

    generated = ""
    if model_choice == "inference_provider":
        try:
            response_text = client.text_generation(
                model="meta-llama/Llama-3.2-3B-Instruct",
                prompt=conversation_prompt,
                max_new_tokens=200,
                temperature=temperature
            )
            generated = response_text.strip()
            logger.info(f"Inference Provider Response:\n{generated}")
        except Exception as e:
            logger.error(f"Error in inference provider: {e}, switching to local model.")
    
    if not generated and local_tokenizer and local_model:
        try:
            # Ensure the tokenizer has a padding token
            if local_tokenizer.pad_token is None:
                local_tokenizer.pad_token = local_tokenizer.eos_token
            
            local_inputs = local_tokenizer(
                conversation_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=400
            )
            local_outputs = local_model.generate(
                input_ids=local_inputs.input_ids,
                attention_mask=local_inputs.attention_mask,
                max_new_tokens=100,  # Controls the length of the generated output
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                pad_token_id=local_tokenizer.eos_token_id
            )
            generated = local_tokenizer.decode(local_outputs[0], skip_special_tokens=True).strip()
            logger.info(f"Local Model Response:\n{generated}")
        except Exception as e:
            logger.error(f"Error in local inference: {e}")
            return "I'm sorry, I'm having trouble understanding you right now."

    assistant_reply = generated.strip()

    # Ensure only the assistant's response is extracted
    if "Assistant:" in assistant_reply:
        assistant_reply = assistant_reply.split("Assistant:", 1)[-1].strip()

    # Remove unnecessary new conversation markers if present
    if "### New Conversation:" in assistant_reply:
        assistant_reply = assistant_reply.split("### New Conversation:")[0].strip()

    logger.info(f"Final Extracted Assistant Reply:\n{assistant_reply}")

    return assistant_reply

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

@app.get("/Overview", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("Overview.html", {"request": request})

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    """Chat API endpoint."""
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
