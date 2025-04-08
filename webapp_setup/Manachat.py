import os
import torch # type: ignore
import logging
from error_logger import log_error
from dotenv import load_dotenv # type: ignore
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from huggingface_hub import InferenceClient # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Externalized configuration via environment variables
try:
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
    MENTAL_MODEL_NAME = os.getenv("MENTAL_MODEL_NAME", "mental/mental-roberta-base")
    LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", 100))
    INFERENCE_MAX_NEW_TOKENS = int(os.getenv("INFERENCE_MAX_NEW_TOKENS", 400))
    LOCAL_TOP_K = int(os.getenv("LOCAL_TOP_K", 50))
    LOCAL_TOP_P = float(os.getenv("LOCAL_TOP_P", 0.95))
    LOCAL_NO_REPEAT_NGRAM_SIZE = int(os.getenv("LOCAL_NO_REPEAT_NGRAM_SIZE", 2))
    LOCAL_TRUNCATION_LENGTH = int(os.getenv("LOCAL_TRUNCATION_LENGTH", 400))
except Exception as e:
    error_message = f"Error loading environment variables: {str(e)}"
    logger.exception(error_message)
    log_error("Manachat.py", "Error", error_message, -1)

# Load environment variables from .env file
load_dotenv()
# Validate Hugging Face credentials
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token is missing. Set the environment variable HUGGINGFACE_TOKEN.")

#HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
#if not HF_INFERENCE_API_KEY:
    #raise ValueError("Inference API key is missing. Set the environment variable HF_INFERENCE_API_KEY.")

# Initialize the InferenceClient for cloud-based inference
client = InferenceClient(api_key=HF_TOKEN)

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
            error_message = f"Failed to load local fallback model: {str(e)}"
            logger.exception(error_message)
            log_error("Manachat.py", "Error", error_message, -1)
            raise RuntimeError("Local model not available")
    return local_model, local_tokenizer

# Load mental health model at startup (always loaded)
try:
    logger.info("Loading mental health model...")
    mental_tokenizer = AutoTokenizer.from_pretrained(MENTAL_MODEL_NAME, token=HF_TOKEN)
    mental_model = AutoModelForMaskedLM.from_pretrained(MENTAL_MODEL_NAME, token=HF_TOKEN)
    logger.info("Mental health model loaded successfully.")
except Exception as e:
    error_message = f"Failed to load mental health model: {str(e)}"
    logger.exception(error_message)
    log_error("Manachat.py", "Error", error_message, -1)
    raise RuntimeError(f"Failed to load mental health model: {e}")

def detect_mental_state(user_message: str) -> str:
    """Detect mental state using the mental health model."""
    try:
        inputs = mental_tokenizer(user_message, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = mental_model(**inputs)
        logits = outputs.logits[:, :-1, :].mean(dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_token_id = torch.argmax(probs, dim=-1)
        mental_state = mental_tokenizer.decode(predicted_token_id)
        return mental_state
    except Exception as e:
        error_message = f"Error detecting mental state: {str(e)}"
        logger.error(error_message)
        log_error("Manachat.py", "Error", error_message, -1)
        return "Unable to detect mental state."

def get_chatbot_response(mental_state: str, user_message: str, 
                         system_prompt: str, model_name: str, 
                         temperature: float) -> str:
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
            local_model_loaded, local_tokenizer_loaded = load_local_model()
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
        error_message = f"Error in {model_name} inference: {str(e)}"
        logger.exception(error_message)
        log_error("Manachat.py", "Error", error_message, -1)
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
