"""
FastAPI Application for Spam Detection & Scam Simulation

Endpoints:
- POST /analyze_message - Analyze a message for spam/scam
- POST /simulate_conversation - Simulate scam conversation
- GET /risk_report - Get risk report for last analysis
"""

import os
import sys
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_message
from src.features import extract_features, get_feature_vector
from src.extract_intel import extract_threat_intelligence
from src.risk_engine import calculate_risk_score, assess_message_risk
from src.scam_baiting_bot import ScamBaitingBot, create_bot, PERSONAS


# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


# Global model cache
class ModelCache:
    spam_vectorizer = None
    spam_model = None
    scam_vectorizer = None
    scam_model = None
    last_analysis = None


def load_models():
    """Load ML models into memory."""
    try:
        spam_vec_path = os.path.join(MODEL_DIR, 'spam_vectorizer.joblib')
        spam_model_path = os.path.join(MODEL_DIR, 'spam_detector.joblib')
        scam_vec_path = os.path.join(MODEL_DIR, 'scam_vectorizer.joblib')
        scam_model_path = os.path.join(MODEL_DIR, 'scam_classifier.joblib')
        
        if os.path.exists(spam_vec_path) and os.path.exists(spam_model_path):
            ModelCache.spam_vectorizer = joblib.load(spam_vec_path)
            ModelCache.spam_model = joblib.load(spam_model_path)
            print("✓ Spam detection models loaded")
        else:
            print("⚠ Spam models not found. Run training first.")
        
        if os.path.exists(scam_vec_path) and os.path.exists(scam_model_path):
            ModelCache.scam_vectorizer = joblib.load(scam_vec_path)
            ModelCache.scam_model = joblib.load(scam_model_path)
            print("✓ Scam classification models loaded")
        else:
            print("⚠ Scam models not found. Run training first.")
            
    except Exception as e:
        print(f"⚠ Error loading models: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("=" * 50)
    print("Starting Spam Detection API...")
    print("=" * 50)
    load_models()
    yield
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Spam Detection & Scam Simulation API",
    description="CPU-only AI system for spam detection, scam classification, and threat analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== Request/Response Models =====================

class MessageRequest(BaseModel):
    text: str = Field(..., description="Message text to analyze")
    sender: Optional[str] = Field(None, description="Sender identifier")
    channel: Optional[str] = Field(None, description="Channel (sms, email, etc)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "URGENT! You won $1,000,000! Click http://fakeprize.com NOW!",
                "sender": "+91-9876543210",
                "channel": "sms"
            }
        }


class SimulationRequest(BaseModel):
    scam_type: str = Field(..., description="Type of scam to simulate")
    persona: str = Field("working_professional", description="Victim persona")
    initial_message: Optional[str] = Field(None, description="Initial scam message")
    num_turns: int = Field(3, ge=1, le=10, description="Number of conversation turns")
    
    class Config:
        json_schema_extra = {
            "example": {
                "scam_type": "bank_phishing",
                "persona": "elderly",
                "initial_message": "URGENT: Your bank account is compromised!",
                "num_turns": 3
            }
        }


class AnalysisResponse(BaseModel):
    is_spam: bool
    spam_confidence: float
    scam_type: Optional[str]
    scam_confidence: Optional[float]
    risk_score: int
    risk_level: str
    recommended_action: str
    threat_intelligence: dict
    features: dict
    processing_time_ms: float


class SimulationResponse(BaseModel):
    scam_type: str
    persona: str
    persona_details: dict
    conversation: list
    risk_assessment: dict
    simulation_stats: dict


class RiskReportResponse(BaseModel):
    has_analysis: bool
    analysis: Optional[dict]
    message: str


# ===================== API Endpoints =====================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Spam Detection & Scam Simulation API",
        "version": "1.0.0",
        "models_loaded": {
            "spam_detector": ModelCache.spam_model is not None,
            "scam_classifier": ModelCache.scam_model is not None
        }
    }


@app.post("/analyze_message", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_message(request: MessageRequest):
    """
    Analyze a message for spam, scam type, and threat intelligence.
    
    Returns:
    - Spam detection result with confidence
    - Scam type classification (if spam)
    - Extracted threat intelligence (URLs, phones, orgs)
    - Risk score (0-100)
    - Recommended action
    """
    start_time = time.time()
    
    # Check if models are loaded
    if ModelCache.spam_model is None:
        raise HTTPException(
            status_code=503,
            detail="Spam detection model not loaded. Run training first: python src/train_spam.py"
        )
    
    text = request.text
    metadata = {
        'sender': request.sender,
        'channel': request.channel
    }
    
    # Preprocess message
    preprocessed = preprocess_message(text, metadata)
    
    # Extract features
    features = extract_features(text)
    
    # Spam detection
    X_spam = ModelCache.spam_vectorizer.transform([preprocessed['processed_text']])
    spam_pred = ModelCache.spam_model.predict(X_spam)[0]
    spam_proba = ModelCache.spam_model.predict_proba(X_spam)[0]
    is_spam = bool(spam_pred)
    spam_confidence = float(spam_proba[1] if is_spam else spam_proba[0])
    
    # Scam type classification (only if spam)
    scam_type = None
    scam_confidence = None
    
    if is_spam and ModelCache.scam_model is not None:
        X_scam = ModelCache.scam_vectorizer.transform([preprocessed['processed_text']])
        scam_type = ModelCache.scam_model.predict(X_scam)[0]
        scam_proba = ModelCache.scam_model.predict_proba(X_scam)[0]
        scam_confidence = float(max(scam_proba))
    
    # Extract threat intelligence
    threat_intel = extract_threat_intelligence(text)
    
    # Calculate risk score
    risk_result = calculate_risk_score(
        spam_confidence=spam_proba[1],  # Always use spam probability
        scam_type=scam_type,
        urgency_score=features.get('urgency_normalized', 0),
        threat_indicator_count=threat_intel.get('threat_indicator_count', 0),
        persona_vulnerability=0.5,  # Default
        has_urls=threat_intel.get('threat_indicators', {}).get('has_urls', False),
        has_phone_numbers=threat_intel.get('threat_indicators', {}).get('has_phones', False),
        has_credential_request=threat_intel.get('threat_indicators', {}).get('has_credential_request', False)
    )
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000
    
    # Store for risk report
    ModelCache.last_analysis = {
        'text': text[:100] + '...' if len(text) > 100 else text,
        'is_spam': is_spam,
        'spam_confidence': spam_confidence,
        'scam_type': scam_type,
        'scam_confidence': scam_confidence,
        'risk_score': risk_result['risk_score'],
        'risk_level': risk_result['risk_level'],
        'recommended_action': risk_result['recommended_action'],
        'action_details': risk_result['action_details'],
        'threat_intelligence': threat_intel,
        'features': features,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return AnalysisResponse(
        is_spam=is_spam,
        spam_confidence=round(spam_confidence, 4),
        scam_type=scam_type,
        scam_confidence=round(scam_confidence, 4) if scam_confidence else None,
        risk_score=risk_result['risk_score'],
        risk_level=risk_result['risk_level'],
        recommended_action=risk_result['recommended_action'],
        threat_intelligence=threat_intel,
        features=features,
        processing_time_ms=round(processing_time, 2)
    )


@app.get("/risk_report", response_model=RiskReportResponse, tags=["Reports"])
async def get_risk_report():
    """
    Get the risk report for the last analyzed message.
    """
    if ModelCache.last_analysis is None:
        return RiskReportResponse(
            has_analysis=False,
            analysis=None,
            message="No message analyzed yet. Use POST /analyze_message first."
        )
    
    return RiskReportResponse(
        has_analysis=True,
        analysis=ModelCache.last_analysis,
        message="Risk report for last analyzed message"
    )


@app.get("/personas", tags=["Reference"])
async def list_personas():
    """Get available victim personas for LLM-powered chat."""
    return {
        "personas": list(PERSONAS.keys()),
        "details": {k: {
            'name': v.name,
            'age': v.age,
            'occupation': v.occupation,
            'background': v.background[:100] + '...'
        } for k, v in PERSONAS.items()}
    }


@app.get("/scam_types", tags=["Reference"])
async def list_scam_types():
    """Get available scam types."""
    return {
        "scam_types": [
            {"type": "bank_phishing", "description": "Fake bank alerts requesting credentials"},
            {"type": "otp_scam", "description": "Tricks to steal OTP codes"},
            {"type": "job_fraud", "description": "Fake job offers requiring fees"},
            {"type": "crypto_scam", "description": "Cryptocurrency investment fraud"},
            {"type": "lottery_scam", "description": "Fake lottery/prize winning notifications"},
            {"type": "loan_scam", "description": "Fraudulent loan offers"},
            {"type": "other", "description": "Other types of scams"}
        ]
    }


# ===================== Scam Baiting Chat Endpoints =====================

# Store active chat sessions
chat_sessions: dict[str, ScamBaitingBot] = {}


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID for conversation")
    scammer_message: str = Field(..., description="Message from the scammer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_001",
                "scammer_message": "Hello sir, this is SBI bank. Your account is blocked. Share OTP to unblock."
            }
        }


class ChatResponse(BaseModel):
    victim_response: str
    persona: str
    mode: str
    scam_type: str
    turn: int
    extracted_intel: dict
    tips: list


@app.post("/chat/start", tags=["Scam Baiting Chat"])
async def start_chat_session(session_id: str = "default"):
    """
    Start a new scam baiting chat session.
    
    The bot will pretend to be a naive victim to:
    - Waste the scammer's time
    - Extract useful information (URLs, phone numbers, etc.)
    - Adapt its persona based on detected scam type
    """
    chat_sessions[session_id] = create_bot()
    return {
        "session_id": session_id,
        "status": "started",
        "message": "Chat session started. Send scam messages to /chat/respond and the bot will play an naive victim."
    }


@app.post("/chat/respond", response_model=ChatResponse, tags=["Scam Baiting Chat"])
async def chat_respond(request: ChatRequest):
    """
    Send a scammer message and get the victim bot's response.
    
    The bot analyzes the scam type and:
    - Selects an appropriate victim persona
    - Generates responses to keep scammer busy
    - Occasionally tries to extract useful info
    - Stalls with realistic delays and excuses
    """
    session_id = request.session_id
    
    # Create session if not exists
    if session_id not in chat_sessions:
        chat_sessions[session_id] = create_bot()
    
    bot = chat_sessions[session_id]
    result = bot.generate_response(request.scammer_message)
    
    # Get scam type from analysis
    scam_type = result.get('scam_analysis', {}).get('scam_type', 'unknown')
    
    # Generate tips based on scam type
    tips = []
    if scam_type == 'otp_scam':
        tips = ["Never share OTP with anyone", "Banks never ask for OTP over phone"]
    elif scam_type == 'bank_phishing':
        tips = ["Verify by calling official bank number", "Banks never ask for credentials over phone"]
    elif scam_type == 'lottery_scam':
        tips = ["Real lotteries don't ask for fees", "If you didn't enter, you can't win"]
    else:
        tips = ["When in doubt, hang up and verify", "Never share personal info with callers"]
    
    return ChatResponse(
        victim_response=result['response'],
        persona=result['persona'],
        mode=result['mode'],
        scam_type=scam_type,
        turn=result['turn'],
        extracted_intel=result['extracted_intel'],
        tips=tips
    )


@app.get("/chat/summary/{session_id}", tags=["Scam Baiting Chat"])
async def get_chat_summary(session_id: str):
    """
    Get summary of a chat session including extracted intelligence.
    """
    if session_id not in chat_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Start a session first with POST /chat/start"
        )
    
    bot = chat_sessions[session_id]
    return bot.get_conversation_summary()


@app.delete("/chat/end/{session_id}", tags=["Scam Baiting Chat"])
async def end_chat_session(session_id: str):
    """
    End a chat session and get final summary.
    """
    if session_id not in chat_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found."
        )
    
    bot = chat_sessions[session_id]
    summary = bot.get_conversation_summary()
    del chat_sessions[session_id]
    
    return {
        "status": "ended",
        "session_id": session_id,
        "summary": summary
    }


# ===================== Main Entry Point =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

