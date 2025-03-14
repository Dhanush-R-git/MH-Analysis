from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import openai  # For free GPT model response (use an API key if required)
from huggingface_hub import login
import os

# Load Hugging Face token from environment variable
os.environ["HUGGINGFACE_TOKEN"] = "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"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise ValueError("Hugging Face token is missing. Set the environment variable HUGGINGFACE_TOKEN.")

app = Flask(__name__, template_folder="/webapp_setup/templates")

# Load mental health model
try:
    tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base", use_auth_token=True)
    model = AutoModelForMaskedLM.from_pretrained("mental/mental-roberta-base", use_auth_token=True)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

def detect_mental_state(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_token_id = torch.argmax(logits, dim=-1)
    mental_state = tokenizer.decode(predicted_token_id[0])
    return mental_state

def get_chatbot_response(mental_state, user_message):
    prompt = f"User is experiencing {mental_state}. Respond supportively: {user_message}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return "I'm sorry, but I couldn't generate a response at this moment."

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    
    mental_state = detect_mental_state(user_message)
    chatbot_response = get_chatbot_response(mental_state, user_message)
    return jsonify({"mental_state": mental_state, "response": chatbot_response})

@app.route("/")
def home():
    return render_template("/workspaces/MHRoberta-a-LLM-for-mental-health-analysis/webapp_setup/templates/web.html")

if __name__ == "__main__":
    app.run(debug=True)

