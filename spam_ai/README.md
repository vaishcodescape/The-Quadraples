# Adaptive Spam Detection & Scam Simulation System

A CPU-only, local AI system for spam detection, scam classification, threat intelligence extraction, and offline scam simulation.

## Features

- **Spam Detection**: TF-IDF + Logistic Regression binary classifier
- **Scam Type Classification**: Identifies bank_phishing, otp_scam, job_fraud, crypto_scam, lottery_scam, loan_scam
- **Threat Intelligence**: Extracts URLs, phone numbers, money amounts, organization names
- **Conversation Simulation**: Offline scam progression simulation
- **Risk Scoring**: 0-100 risk score with recommendations

## Requirements

- Python 3.10+
- CPU only (no GPU required)
- Windows/Linux

## Installation

```bash
cd spam_ai
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Training Models

```bash
python src/train_spam.py
python src/train_scam_type.py
```

## Running the API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Access Swagger UI at: http://127.0.0.1:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze_message` | POST | Analyze message for spam/scam |
| `/simulate_conversation` | POST | Simulate scam conversation |
| `/risk_report` | GET | Get risk report for session |

## Example Usage

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/analyze_message",
    json={"text": "URGENT! You won $1,000,000! Click http://fakeprize.com NOW!"}
)
print(response.json())
```

## Project Structure

```
spam_ai/
├── data/             # Training datasets
├── models/           # Trained model files
├── src/
│   ├── api.py        # FastAPI application
│   ├── preprocess.py # Message preprocessing
│   ├── features.py   # Feature engineering
│   ├── train_spam.py # Spam model training
│   ├── train_scam_type.py # Scam classifier training
│   ├── extract_intel.py   # Threat intelligence
│   ├── simulate_chat.py   # Conversation simulation
│   └── risk_engine.py     # Risk scoring
├── requirements.txt
└── README.md
```

## Ethical Notice

This system is designed for **educational and defensive purposes only**:
- No real attacker engagement
- No tracking or doxxing
- Offline simulation only
- No impersonation of real users
