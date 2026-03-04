"""
api/main.py — FastAPI Gateway

Serves the CANARY ML models and Business Rules Engine via a standard REST API.
This is the entry point for downstream legacy banking systems to consume the AI.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import time

from inference import FraudDetector, get_detector
from aml.aml_inference import AMLDetector
from api.rules import evaluate_fraud_rules, override_ml_decision


app = FastAPI(
    title="CANARY Enterprise API",
    description="Contextual Anomaly Network for Anomaly Recognition & Analysis",
    version="1.0.0"
)

# CORS for the Next.js Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy-loaded Models ────────────────────────────────────────────────────────
# Load models lazily to avoid crashing if checkpoints aren't available yet

_fraud_detector: Optional[FraudDetector] = None
_aml_detector: Optional[AMLDetector] = None

def get_fraud_model() -> FraudDetector:
    global _fraud_detector
    if _fraud_detector is None:
        try:
            _fraud_detector = get_detector("checkpoints/best_model.pt")
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Fraud model checkpoint not found. Train model first.")
    return _fraud_detector


def get_aml_model() -> AMLDetector:
    global _aml_detector
    if _aml_detector is None:
        try:
            _aml_detector = AMLDetector.from_checkpoint("checkpoints/aml_best.pt")
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="AML model checkpoint not found. Train model first.")
    return _aml_detector


# ── API Models ────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    merchant_cat: str
    country: str
    is_domestic: bool
    hour: int
    day_of_week: int
    channel: str
    currency: str
    velocity: str = "NORMAL"
    flags: List[str] = []

class FraudResponse(BaseModel):
    transaction_id: str
    timestamp: float
    decision_source: str
    risk_label: str
    action: str
    fraud_probability: float
    confidence: str
    risk_factors: List[str]
    processing_time_ms: float

class AMLRequest(BaseModel):
    account_id: str
    transactions: List[Dict[str, Any]] = Field(..., description="List of up to 30 recent transactions")

class AMLResponse(BaseModel):
    account_id: str
    aml_probability: float
    is_suspicious: bool
    risk_label: str
    action: str
    signals: List[str]
    n_transactions: int
    processing_time_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "models": {
            "fraud": os.path.exists("checkpoints/best_model.pt"),
            "aml": os.path.exists("checkpoints/aml_best.pt"),
        }
    }


@app.post("/v1/score/fraud", response_model=FraudResponse)
async def score_fraud(req: TransactionRequest):
    """
    Score a single transaction in real-time.
    Executes Business Rules Engine first, then ML model, and coalesces the decision.
    """
    t0 = time.time()

    # 1. Convert Pydantic request to dict
    txn_dict = req.model_dump()

    # 2. Run Business Rules Engine
    rule_label, rule_action, rule_name = evaluate_fraud_rules(txn_dict)

    # 3. Run ML Model
    model = get_fraud_model()
    ml_result = model.predict(txn_dict)

    # 4. Merge Decisions
    final_label, final_action, decision_source = override_ml_decision(
        ml_label=ml_result["risk_label"],
        ml_action=ml_result["action"],
        ml_prob=ml_result["fraud_probability"],
        rule_label=rule_label,
        rule_action=rule_action,
        rule_name=rule_name
    )

    processing_time_ms = (time.time() - t0) * 1000

    return FraudResponse(
        transaction_id=req.transaction_id,
        timestamp=time.time(),
        decision_source=decision_source,
        risk_label=final_label,
        action=final_action,
        fraud_probability=ml_result["fraud_probability"],
        confidence=ml_result["confidence"],
        risk_factors=ml_result["risk_factors"],
        processing_time_ms=round(processing_time_ms, 2)
    )


@app.post("/v1/score/aml", response_model=AMLResponse)
async def score_aml(req: AMLRequest):
    """
    Score an account's recent transaction history for money laundering patterns.
    """
    t0 = time.time()
    model = get_aml_model()

    result = model.score_account(req.transactions)

    processing_time_ms = (time.time() - t0) * 1000

    return AMLResponse(
        account_id=req.account_id,
        aml_probability=result["aml_probability"],
        is_suspicious=result["is_suspicious"],
        risk_label=result["risk_label"],
        action=result["action"],
        signals=result["signals"],
        n_transactions=result["n_transactions"],
        processing_time_ms=round(processing_time_ms, 2)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
