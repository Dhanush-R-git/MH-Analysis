from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore
from pydantic import BaseModel, Field # type: ignore
import uvicorn # type: ignore
from typing import List, Dict, Optional
import asyncio
import logging
import os

# Import backend functions
from Manachat import detect_mental_state, get_chatbot_response, load_local_model, HF_INFERENCE_API_KEY
from Mananow import QUESTION_FLOW, generate_mentanow_report, build_mentanow_prompt, analyze_comments_sentiment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app and Jinja2 templates
app = FastAPI(title="Maṉa-Mental Health Analysis")
app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./webapp_setup/templates")

# Define Pydantic model for chat requests
class ChatRequest(BaseModel):
    user_message: str
    system_prompt: str
    model_name: str
    temperature: float

class MentaNowRequest(BaseModel):
    userAnswer: Optional[str] = Field(None, alias="userAnswer")
    userAnswers: List[Dict] = Field(default_factory=list, alias="userAnswers")
    
    class Config:
        allow_population_by_field_name = True

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

@app.get("/Research", response_class=HTMLResponse)
async def overview_page(request: Request):
    return templates.TemplateResponse("Research.html", {"request": request})

@app.get("/Overview", response_class=HTMLResponse)
async def overview_page(request: Request):
    return templates.TemplateResponse("Overview.html", {"request": request})

# New endpoint to process file uploads for sentiment analysis
@app.post("/api/upload-comments")
async def upload_comments(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        logger.info(f"Received file: {file.filename}")
        logger.info(f"File size: {len(file_bytes)} bytes")
        json_data = file_bytes.decode("utf-8")
        logger.info(f"Decoded JSON data (first 200 chars): {json_data[:200]}")
        sentiment_counts = analyze_comments_sentiment(json_data)
        logger.info(f"Sentiment counts: {sentiment_counts}")
        # If an error key is present in sentiment_counts, return a 400 status.
        if "error" in sentiment_counts:
            return JSONResponse(status_code=400, content=sentiment_counts)
        return JSONResponse(status_code=200, content=sentiment_counts)
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Failed to process uploaded file."})



@app.get("/Manachat", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    file_path = "D:/project-main/Final-year-projects/Project-Laboratory/MH-Analysis/webapp_setup/templates/Manachat.html"
    logger.info(f"Checking if file exists: {file_path}")
    assert os.path.exists(file_path), f"Template file not found at {file_path}"
    return templates.TemplateResponse("Manachat.html", {"request": request})

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
        
        # Offload blocking model inference calls to background threads
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
    
# Endpoints for MentaNow UI
@app.get("/ManaNow", response_class=HTMLResponse)
async def mana_now_page(request: Request):
    return templates.TemplateResponse("ManaNow.html", {"request": request})


@app.post("/api/ManaNow")
def mentanow_api(payload: MentaNowRequest):
    """
    Returns either the next question or a final 'report' based on the user's answers.
    """
    user_answers = payload.userAnswers  # now correctly populated via alias
    current_index = len(user_answers)
    total_questions = len(QUESTION_FLOW)

    if current_index < total_questions:
        next_question = QUESTION_FLOW[current_index].copy()
        next_question.update({"number": current_index + 1, "total": total_questions})
        return JSONResponse(content=next_question)
    
    # All questions answered: generate final report.
    report = generate_mentanow_report(user_answers)
    return JSONResponse(content={
        "status": "complete",
        "report": report,
        "resources": [
            {"name": "Crisis Hotline", "contact": "1-800-273-TALK"},
            {"name": "Mental Health America", "url": "https://mhanational.org"}
        ]
    })

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="127.0.0.1", port=5000, reload=True)