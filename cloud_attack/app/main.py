import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.generate import run_evaluation, MODEL_REGISTRY

app = FastAPI(title="Adversarial MNIST Evaluation")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


class EvaluationRequest(BaseModel):
    source_model: str
    num_examples: int = 20


class EvaluationResponse(BaseModel):
    source_model: str
    vision_model: str
    total_examples: int
    correct: int
    accuracy: float
    results: list


@app.post("/generate", response_model=EvaluationResponse)
def generate_and_evaluate(request: EvaluationRequest):
    return run_evaluation(request.source_model, GEMINI_API_KEY, request.num_examples)