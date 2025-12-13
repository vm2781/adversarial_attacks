import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.generate import run_evaluation,MODEL_REGISTRY
from app.imagenet import run_evaluation as run_imagenette_evaluation

app = FastAPI(title="Adversarial MNIST Evaluation")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class EvaluationRequest(BaseModel):
    source_model: str
    attack_type: str
    num_examples: int = 20


class EvaluationResponse(BaseModel):
    source_model: str
    attack_type: str
    vision_model: str
    total_examples: int
    correct: int
    accuracy: float
    results: list

class ImageNetteRequest(BaseModel):
    attack_type: str
    seed: int
    num_examples: int = 20


class ImageNetteResponse(BaseModel):
    dataset: str
    source_model: str
    attack_type: str
    vision_model: str
    total_tested: int
    correct: int
    accuracy: float
    results: list


@app.post("/generate", response_model=EvaluationResponse)
def generate_and_evaluate(request: EvaluationRequest):
    if request.source_model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable not set"
        )
    
    if request.num_examples < 1 or request.num_examples > 500:
        raise HTTPException(
            status_code=400,
            detail="num_examples must be between 1 and 500"
        )
    
    return run_evaluation(request.source_model, request.attack_type, GEMINI_API_KEY, request.num_examples)

@app.post("/imagenette", response_model=ImageNetteResponse)
def imagenette_evaluate(request: ImageNetteRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable not set"
        )
    
    if request.num_examples < 1 or request.num_examples > 500:
        raise HTTPException(
            status_code=400,
            detail="num_examples must be between 1 and 500"
        )
    
    return run_imagenette_evaluation(request.attack_type, GEMINI_API_KEY, request.seed, request.num_examples)


@app.get("/health")
def health():
    return {"status": "healthy", "available_models": list(MODEL_REGISTRY.keys())}