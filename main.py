from fastapi import FastAPI
from schemas import AnalysisRequest, AnalysisResponse
from model import MODEL_BACKEND, analyze_sentiment

app = FastAPI(title="Sentiment Analysis API")

@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API is running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_backend": MODEL_BACKEND}

# @app.post("/analyze", response_model=AnalysisResponse)
# def get_sentiment(request: AnalysisRequest):
#     # Call our model function
#     prediction = analyze_sentiment(request.text)
    
#     return {
#         "label": prediction["label"],
#         "score": prediction["score"]
#     }
@app.post("/analyze")
def get_sentiment(request: AnalysisRequest):
    prediction = analyze_sentiment(request.text)
    
    label = prediction["label"]
    score = prediction["score"]
    
    # Logic for Neutral: if the AI isn't very sure (e.g., < 60% confidence)
    if score < 0.60:
        label = "NEUTRAL"
        
    return {"label": label, "score": score}