from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A REST API for sentiment classification using DistilBERT",
    version="1.0.0"
)

# Load model at startup
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Request/Response schemas
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    
    class Config:
        json_schema_extra = {
            "example": {"text": "This movie was absolutely fantastic!"}
        }

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., max_length=100)

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str

# Endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and model is loaded."""
    return {
        "status": "healthy",
        "model": model_name,
        "device": str(device)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    """Predict sentiment for a single text."""
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        
        return {
            "label": model.config.id2label[pred_idx],
            "confidence": probs[pred_idx].item(),
            "probabilities": {
                model.config.id2label[i]: probs[i].item() 
                for i in range(len(probs))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchRequest):
    """Predict sentiment for multiple texts."""
    results = []
    for text in request.texts:
        req = PredictionRequest(text=text)
        results.append(predict_sentiment(req))
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)