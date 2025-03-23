from fastapi import FastAPI, Request, Response # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore
from pydantic import BaseModel # type: ignore
import uvicorn # type: ignore
import asyncio
import logging

# Import backend functions
from Manachat import detect_mental_state, get_chatbot_response, load_local_model, HF_INFERENCE_API_KEY
from Mananow import QUESTION_FLOW, generate_mentanow_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="D:/project-main/Final-year-projects/Project-Laboratory/MH-Analysis/webapp_setup/templates")

# Define Pydantic model for chat requests
class ChatRequest(BaseModel):
    user_message: str
    system_prompt: str
    model_name: str
    temperature: float
# Pydantic model for the MentaNow request payload
class MentaNowRequest(BaseModel):
    userAnswer: str = None
    userAnswers: list = None 

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

@app.get("/Overview", response_class=HTMLResponse)
async def overview_page(request: Request):
    return templates.TemplateResponse("Overview.html", {"request": request})

@app.get("/manachat", response_class=HTMLResponse)
async def chatbot_page(request: Request):
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
@app.get("/mananow", response_class=HTMLResponse)
async def mentanow_page(request: Request):
    return templates.TemplateResponse("ManaNow.html", {"request": request})

@app.post("/api/ManaNow")
def mentanow_api(payload: MentaNowRequest):
    """
    MentaNow API endpoint: returns either the next question in the flow or, if complete,
    returns a final report summarizing the user's responses.
    """
    user_answers = payload.userAnswers or []
    current_index = len(user_answers)
    if current_index < len(QUESTION_FLOW):
        next_question = QUESTION_FLOW[current_index]
        return JSONResponse(content=next_question)
    
    final_report = generate_mentanow_report(user_answers)
    return JSONResponse(content={
        "type": "report",
        "reportText": final_report
    })

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="127.0.0.1", port=8000, reload=True)

