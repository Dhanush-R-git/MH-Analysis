import os
import logging
from huggingface_hub import InferenceClient  # type: ignore
from dotenv import load_dotenv  # type: ignore
from typing import List, Dict
import json
from fastapi.responses import JSONResponse  # type: ignore

# New imports for sentiment analysis
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np  # type: ignore
from scipy.special import softmax  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
HUGGINGFACE_TOKEN1 = os.getenv("HUGGINGFACE_TOKEN1")
if not HUGGINGFACE_TOKEN1:
    raise ValueError("Hugging Face token is missing. Set the environment variable")
if not HF_INFERENCE_API_KEY:
    raise ValueError("HF_INFERENCE_API_KEY missing. Please set it.")



mentanow_client = InferenceClient(
    provider="together",
    api_key=os.getenv("HF_INFERENCE_API_KEY")  # Ensure this token has read permissions
)


# Extended question flow (10 questions)
QUESTION_FLOW = [
    {
        "id": "user_type",
        "title": "Are you answering for yourself or someone else?",
        "text": "Please select one option.",
        "type": "radio",
        "options": ["Myself", "Someone else"]
    },
    {
        "id": "age_range",
        "title": "What is your age range?",
        "text": "Choose your age range.",
        "type": "radio",
        "options": ["Under 18", "18-30", "31-50", "50+"]
    },
    {
        "id": "primary_concern",
        "title": "What is your primary concern right now?",
        "text": "Briefly describe what is troubling you (e.g., stress, anxiety).",
        "type": "text"
    },
    {
        "id": "sleep",
        "title": "Have you experienced any sleep disturbances recently?",
        "text": "Please describe if you have had issues with sleep.",
        "type": "text"
    },
    {
        "id": "concentration",
        "title": "Do you find it difficult to concentrate?",
        "text": "Rate your ability to focus on tasks.",
        "type": "radio",
        "options": ["Not at all", "Somewhat", "Very much"]
    },
    {
        "id": "energy",
        "title": "How would you rate your energy levels?",
        "text": "Select one option.",
        "type": "radio",
        "options": ["High", "Moderate", "Low"]
    },
    {
        "id": "social",
        "title": "Have you withdrawn from social activities?",
        "text": "Please select one option.",
        "type": "radio",
        "options": ["Yes", "No"]
    },
    {
        "id": "appetite",
        "title": "How has your appetite been?",
        "text": "Choose your response.",
        "type": "radio",
        "options": ["Increased", "Decreased", "Normal"]
    },
    {
        "id": "mood",
        "title": "How would you describe your overall mood lately?",
        "text": "Please select the option that best describes your mood.",
        "type": "radio",
        "options": ["Happy", "Neutral", "Sad"]
    },
    {
        "id": "coping",
        "title": "How effective do you feel your current coping strategies are?",
        "text": "Rate your ability to manage stress.",
        "type": "radio",
        "options": ["Effective", "Somewhat effective", "Not effective"]
    }
]

def build_mentanow_prompt(user_answers: List[Dict]) -> str:
    """Construct a structured prompt summarizing the user's answers."""
    summary = "\n".join(
        f"Q{i+1}: {ans['question']}\nA{i+1}: {ans['answer']}\n"
        for i, ans in enumerate(user_answers)
    )
    return f"""Mental Health Assessment Summary:
{summary}

Please provide:
1. A concise analysis of the user's mental state.
2. Three actionable recommendations.
3. Suggestions for professional help if needed.

Guidelines:
- Use empathetic, non-judgmental language.
- Avoid medical terminology.
- Focus on practical strategies.
- Include crisis resources if warranted."""

def generate_mentanow_report(user_answers: List[Dict]) -> str:
    """Generate assessment report with error handling using meta-llama/Llama-3.2-3B-Instruct."""
    try:
        prompt = build_mentanow_prompt(user_answers)
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Call the inference client using the Together provider and the meta-llama model.
        completion = mentanow_client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=400,
            temperature=0.7,
        )
        
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return "We encountered an error generating your report. Please try again later."



'''def generate_mentanow_report(user_answers: List[Dict]) -> str:
    """Generate assessment report with error handling."""
    try:
        prompt = build_mentanow_prompt(user_answers)
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Using run() for text generation. Adjust parameters as needed.
        response = mentanow_client.run(
            prompt,
            task="text-generation",
            parameters={"max_tokens": 400, "temperature": 0.7}
        )
        # Assuming the response is a dict with a "generated_text" key.
        generated_text = response.get("generated_text", "")
        if not generated_text:
            raise ValueError("Empty response from inference client.")
        return generated_text.strip()
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return "We encountered an error generating your report. Please try again later.'''

# -------- New Code for File Upload and Sentiment Analysis --------

try:
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HUGGINGFACE_TOKEN1)
    config = AutoConfig.from_pretrained(MODEL, token=HUGGINGFACE_TOKEN1)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, token=HUGGINGFACE_TOKEN1)
except Exception as e:
    logger.error(f"Failed to load sentiment analysis model: {str(e)}")
    
def preprocess(text: str) -> str:
    """Preprocess text by replacing usernames and URLs with placeholders."""
    new_text = []
    for t in text.split():
        if t.startswith('@') and len(t) > 1:
            t = '@user'
        elif t.startswith('http'):
            t = 'http'
        new_text.append(t)
    return " ".join(new_text)

def analyze_comments_sentiment(json_data: str) -> dict:
    """
    Analyzes sentiment of comments from uploaded JSON data.
    Expects JSON data in one of the following formats:
      - A JSON array of objects
      - A single JSON object (which will be wrapped into a list)
      - Line-separated JSON objects
    Only the "value" field under "Comment" is used for analysis.
    Returns counts of 'Positive' and 'Negative' sentiments.
    Note: 'Neutral' is treated as 'Negative'.
    """
    data = []
    try:
        # Try to parse the entire input as JSON.
        data = json.loads(json_data)
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("Parsed JSON is not a list or dict.")
    except Exception as e:
        logger.warning("Full JSON parse failed, attempting line-by-line parsing. Error: %s", e)
        # Fallback: Attempt to parse line-separated JSON objects.
        for line in json_data.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove trailing commas if present.
            if line.endswith(','):
                line = line[:-1]
            try:
                item = json.loads(line)
                data.append(item)
            except Exception as ex:
                logger.warning("Skipping line due to error: %s", ex)
                continue

    if not data:
        logger.error("No valid JSON objects found in the uploaded file.")
        return {"error": "No valid JSON objects found in the uploaded file."}

    positive_count = 0
    negative_count = 0

    for item in data:
        try:
            comment_text = item["string_map_data"]["Comment"]["value"]
        except KeyError:
            continue
        
        # Ensure we have a non-empty string.
        if not isinstance(comment_text, str) or not comment_text.strip():
            continue
        
        processed_text = preprocess(comment_text)
        try:
            encoded_input = tokenizer(processed_text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)[::-1]
            top_label = config.id2label[ranking[0]]
        except Exception as ex:
            logger.warning("Error processing comment '%s': %s", comment_text, ex)
            continue
        
        if top_label == "Positive":
            positive_count += 1
        else:
            # Treat both 'Neutral' and 'Negative' as Negative.
            negative_count += 1

    logger.info("Final Sentiment counts: %s", {"positive": positive_count, "negative": negative_count})
    return {"positive": positive_count, "negative": negative_count}
