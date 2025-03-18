# Mental Health Analysis

This project focuses on leveraging machine learning and natural language processing (NLP) techniques to analyze mental health-related text data. It includes a fine-tuned version of the Roberta transformer model, named MHRoberta, specifically designed for mental health analysis tasks. The model is trained using the PEFT (Parameter-Efficient Fine-Tuning) method on a mental health dataset.

## MHRoberta (a Large Language Model for mental health analysis)

We developed our own model called 'MHRoberta' is Mental Health Roberta model. It is pretrained Roberta transformer based model fine-tunned on Mental Health dataset by adopting PEFT method.

### Key Features

- **MHRoberta Model**: A transformer-based model fine-tuned for mental health analysis tasks.
- **Chatbot Integration**: A FastAPI-powered chatbot that interacts with users, detects their mental state, and provides empathetic responses based on the detected state.
- **Local and Cloud Inference**: Supports both local fallback models and cloud-based inference using Hugging Face's Inference API.
- **Mental State Detection**: Automatically detects mental states from user input and tailors responses accordingly.

## create an virtual python environment

```bash
conda create -n env.0.0.0 python=3.13.2 -y
```

```bash
conda activate env.0.0.0
```

## install requirements file

```bash
pip install -r requirement.txt
```

## how to run this project

export this huggingface tokens in terminal

```bash
export HUGGINGFACE_TOKEN=your_huggingface_token
export HF_INFERENCE_API_KEY=your_inference_api_key
```

export the project root in terminal

```bash
export PYTHONPATH=/workspaces/MHRoberta-a-LLM-for-mental-health-analysis
```

run the backend file in terminal

```bash
python webapp_setup/chatbot.py
```