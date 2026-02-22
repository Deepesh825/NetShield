from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from backend.core import analyzer

app = FastAPI(title="NetShield API")

# Enable CORS (still good practice even when unified)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
HTML_FILE_PATH = os.path.join(FRONTEND_DIR, "index.html")

@app.get("/")
async def serve_ui():
    """Serves the HTML frontend to the browser."""
    if not os.path.exists(HTML_FILE_PATH):
        raise HTTPException(status_code=404, detail="Frontend HTML file not found.")
    return FileResponse(HTML_FILE_PATH)

# api endpoint
class URLRequest(BaseModel):
    url: str

@app.post("/api/analyze")
async def analyze_endpoint(request: URLRequest):
    """Receives the URL from the frontend and runs the Analysis."""
    try:
        result = analyzer.analyze_url(request.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))