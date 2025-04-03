import os
import logging
import requests # type: ignore
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

API_URL = "https://router.huggingface.co/together/v1/chat/completions"

HF_TOKEN = os.getenv("HF_INFERENCE_API_KEY")
if not HF_TOKEN:
    raise ValueError("Hugging Face token is missing. Set the environment variable")


'''
try:
    client = InferenceClient(
        model="deepseek-ai/DeepSeek-R1",
        provider="together",
        api_key=HF_INFERENCE_API_KEY  # Ensure this token has read permissions
    )
except Exception as e:
    logger.error(f"Failed to create InferenceClient: {str(e)}")
    logger.info("Please check your API key and model name.")
'''

# Extended question flow (10 questions)
QUESTION_FLOW = [
    {
        "id": "user_name",
        "title": "Please Enter your name?",
        "text": "Please provide your name.",
        "type": "text"
    },
    {
        "id": "user_type",
        "title": "Are you sure answering for yourself ?",
        "text": "Please select one option.",
        "type": "radio",
        "options": ["Yes", "No"]
    },
    {
        "id": "Date_of_Birth",
        "title": "Please Enter your Date of Birth?",
        "text": "Select your date of birth from the calendar.",
        "type": "date"
    },
    {
        "id": "person_age",
        "title": "What is your age ?",
        "type": "number",
        "min": 16,
        "max": 100,
        "step": 1,
        "placeholder": "Enter your age",
    },
    {
        "id": "smoker",
        "title": "Are you a smoker?",
        "text": "Please select one option.",
        "type": "radio",
        "options": ["Yes", "No"],
        "follow_up": {
            "yes": [
                {
                    "id": "cigarettes_per_day",
                    "title": "How many cigarettes do you smoke per day?",
                    "text": "Please enter the number of cigarettes you smoke daily.",
                    "type": "number",
                    "placeholder": "Enter number of cigarettes"
                },
                {
                    "id": "smoking_duration",
                    "title": "How long have you been smoking?",
                    "text": "Please specify the duration of your smoking habit.",
                    "type": "number",
                    "placeholder": "Enter duration",
                    "unit_options": ["Years", "Months", "Days"]
                }
            ]
        }
    },
    {
        "id": "alcohol",
        "title": "Do you consume alcohol?",
        "text": "Please select one option.",
        "type": "radio",
        "options": ["Yes", "No"],
        "follow_up": {
            "yes": [
                {
                    "id": "drinks_per_week",
                    "title": "How many alcoholic drinks do you consume per week?",
                    "text": "Please enter the approximate number of drinks you consume weekly.",
                    "type": "number",
                    "placeholder": "Enter number of drinks"
                },
                {
                    "id": "alcohol_duration",
                    "title": "How long have you been consuming alcohol?",
                    "text": "Please specify the duration of your alcohol consumption habit.",
                    "type": "number",
                    "placeholder": "Enter duration",
                    "unit_options": ["Years", "Months", "Days"]
                }
            ]
        }
    },
    {
        "id": "medications",
        "title": "Are you currently taking any medications?",
        "text": "Please select one option.",
        "type": "radio",
        "options": ["Yes", "No"],
        "follow_up": {
            "yes": [
                {
                    "id": "medication_list",
                    "title": "Please list the medications you are currently taking.",
                    "text": "Provide the names of the medications you are taking (e.g., Paracetamol, Ibuprofen).",
                    "type": "text",
                    "placeholder": "Enter medication names separated by commas"
                },
                {
                    "id": "medication_duration",
                    "title": "How long have you been taking these medications?",
                    "text": "Please specify the duration of your medication usage.",
                    "type": "number",
                    "placeholder": "Enter duration",
                    "unit_options": ["Years", "Months", "Days"]
                },
                {
                    "id": "medication_side_effects",
                    "title": "Have you experienced any side effects from these medications?",
                    "text": "Please describe any side effects you have experienced.",
                    "type": "text",
                    "placeholder": "Describe side effects here..."
                }
            ]
        }
    },
    {
        "id": "sleep",
        "title": "Have you experienced any sleep disturbances recently?",
        "text": "Rate the frequency of your sleep disturbances and provide additional details if applicable.",
        "type": "likert_and_text",  # Custom type to combine Likert scale and text input
        "likert_scale": {
            "1": "Not at all",
            "2": "Rarely",
            "3": "Sometimes",
            "4": "Often",
            "5": "Very often"
        },
        "follow_up": {
            "id": "sleep_details",
            "title": "Please describe your sleep disturbances.",
            "text": "Provide additional details about your sleep issues (e.g., difficulty falling asleep, waking up frequently).",
            "type": "text",
            "placeholder": "Describe your sleep issues here..."
        }
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
        "text": "Appetite refers to your natural desire to eat. Changes in appetite can indicate physical or emotional health changes. Please choose your response.",
        "type": "radio",
        "options": ["Increased", "Decreased", "Normal"],
        "follow_up": {
            "increased": [
                {
                    "id": "increased_appetite_reason",
                    "title": "Do you know why your appetite has increased?",
                    "text": "Please provide any reasons or observations about your increased appetite.",
                    "type": "text",
                    "placeholder": "Enter your response here..."
                }
            ],
            "decreased": [
                {
                    "id": "decreased_appetite_reason",
                    "title": "Do you know why your appetite has decreased?",
                    "text": "Please provide any reasons or observations about your decreased appetite.",
                    "type": "text",
                    "placeholder": "Enter your response here..."
                },
                {
                    "id": "weight_loss",
                    "title": "Have you experienced any weight loss?",
                    "text": "If yes, please specify the amount of weight lost.",
                    "type": "number",
                    "placeholder": "Enter weight loss in kilograms"
                }
            ]
        }
    },
    {
        "id": "mood",
        "title": "How would you describe your overall mood lately?",
        "text": "Please select the option that best describes your mood.",
        "type": "radio",
        "options": ["Happy", "Neutral", "Sad"]
    },
    {
        "id": "relationship_problems",
        "title": "Are you currently experiencing any relationship problems?",
        "text": "Relationship problems can include issues with friends, family, or romantic relationships. Please select one option.",
        "type": "radio",
        "options": ["Yes", "No"],
        "follow_up": {
            "yes": [
                {
                    "id": "relationship_type",
                    "title": "What type of relationship problem are you experiencing?",
                    "text": "Please select the type(s) of relationship issue(s) you are facing. You can choose one or more options.",
                    "type": "checkbox",
                    "options": ["Friends", "Family", "Romantic Partner", "Breakups", "Other"],
                    "follow_up": {
                        "Friends": [
                            {
                                "id": "friends_problem_description",
                                "title": "Please describe the problem with your friends.",
                                "text": "Provide details about the issues you are facing with your friends.",
                                "type": "text",
                                "placeholder": "Describe your problem here..."
                            }
                        ],
                        "Family": [
                            {
                                "id": "family_problem_description",
                                "title": "Please describe the problem with your family.",
                                "text": "Provide details about the issues you are facing with your family.",
                                "type": "text",
                                "placeholder": "Describe your problem here..."
                            }
                        ],
                        "Romantic Partner": [
                            {
                                "id": "romantic_problem_description",
                                "title": "Please describe the problem with your romantic partner.",
                                "text": "Provide details about the issues you are facing with your romantic partner.",
                                "type": "text",
                                "placeholder": "Describe your problem here..."
                            }
                        ],
                        "Breakups": [
                            {
                                "id": "breakup_problem_description",
                                "title": "Please describe the breakup problem you are experiencing.",
                                "text": "Provide details about the breakup and how it is affecting you.",
                                "type": "text",
                                "placeholder": "Describe your problem here..."
                            }
                        ],
                        "Other": [
                            {
                                "id": "other_problem_description",
                                "title": "Please describe the problem you are experiencing.",
                                "text": "Provide details about the issue you are facing.",
                                "type": "text",
                                "placeholder": "Describe your problem here..."
                            }
                        ]
                    }
                }
            ]
        }
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
"GENERATE A Complete REPORT BASED ON MENTAL HEALTH ASSESSMENT SUMMARY OF USER."
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
        logger.info(f"Generated prompt:\n{prompt}")
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "deepseek-ai/DeepSeek-R1",
            "max_tokens": 500,
            "temperature": 0.7
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return "Error generating report. Please try again."

        return response.json()['choices'][0]['message']['content'].strip()
    
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return "We encountered an error generating your report. Please try again later."
'''
        # Call the inference client using the Together provider and the meta-llama model.
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.7,
        )
        logger.info(f"Generated report:\n{completion.choices[0].message.content.strip()}")
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        if "401" in str(e):
            return "Report generation failed due to invalid credentials. Please check your HF_INFERENCE_API_KEY."
        return "We encountered an error generating your report. Please try again later."
'''

# -------- New Code for File Upload and Sentiment Analysis -------- token=HUGGINGFACE_TOKEN1

try:
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
except Exception as e:
    logger.error(f"Failed to load sentiment analysis model: {str(e)}")
    
def preprocess(text: str) -> str:
    """Preprocess text by lowercasing, and replacing usernames and URLs with placeholders."""
    #text = text.lower()  # Convert text to lowercase for consistency.
    new_text = []
    for t in text.split():
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_comments_sentiment(json_data: str) -> dict:
    """
    Analyzes sentiment of comments from uploaded JSON data.
    Expects JSON data in one of the following formats:
      - A JSON array of objects
      - A single JSON object (which will be wrapped into a list)
      - Line-separated JSON objects
    Only the "value" field under "Comment" will be used for analysis.
    Returns counts of 'Positive', 'Negative', and 'Neutral' sentiments along with detailed results.
    """
    data = []
    try:
        data = json.loads(json_data)
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("Parsed JSON is not a list or dict.")
    except Exception as e:
        logger.warning("Full JSON parse failed, attempting line-by-line parsing. Error: %s", e)
        for line in json_data.strip().splitlines():
            line = line.strip()
            if not line:
                continue
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
    neutral_count = 0
    details = []  # Store detailed results

    for item in data:
        try:
            comment_text = item["string_map_data"]["Comment"]["value"]
        except KeyError:
            continue

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
            # Debug log: print scores for each label
            logger.info("Text: '%s' | Scores: %s | Top label: %s", comment_text, scores, top_label)
        except Exception as ex:
            logger.warning("Error processing comment '%s': %s", comment_text, ex)
            continue

        # Count each sentiment category separately
        if top_label == "positive":
            positive_count += 1
            sentiment = "positive"
        elif top_label == "neutral":
            neutral_count += 1
            sentiment = "neutral"
        else:
            negative_count += 1
            sentiment = "negative"

        truncated_text = comment_text if len(comment_text) <= 200 else comment_text[:100] + "..."
        details.append({
            "extracted_text": truncated_text,
            "sentiment": sentiment
        })
    
    logger.info("Final Sentiment counts: %s", {"positive": positive_count, "neutral": neutral_count, "negative": negative_count})
    return {"positive": positive_count, "neutral": neutral_count, "negative": negative_count, "details": details}

# Example test input for analyze_comments_sentiment
#test_json_data = '[{"string_map_data": {"Comment": {"value": "This is a great product!"}}}]'
#print(analyze_comments_sentiment(test_json_data))
