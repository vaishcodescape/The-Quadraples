# Spam & Scam Detection REST API

High-performance REST API for spam detection, scam classification, threat intelligence extraction, **rule-based** scam conversation escalation, and risk scoring. CPU-only, Dockerized, ready for **GCP VM** or **Cloud Run**.

## Features

- **Spam detection** – TF-IDF + Logistic Regression; returns `is_spam` and confidence
- **Scam type classification** – bank_phishing, otp_scam, job_fraud, crypto_scam, lottery_scam, loan_scam, other
- **Threat intelligence** – URLs, phone numbers, money amounts, org names (spaCy NER), action phrases
- **Rule-based escalation simulation** – Offline simulation of N conversation turns (no LLM)
- **Risk score (0–100)** and **recommended action** with details

## Requirements

- Python 3.10+
- CPU only

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Training models

From the project root:

```bash
python -m training.train_spam
python -m training.train_scam_type
```

## Running the API locally

From the project root:

```bash
# Development
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Or with env (production-style)
HOST=0.0.0.0 PORT=8000 uvicorn api.main:app
```

- API: http://127.0.0.1:8000  
- Docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  
- Readiness: http://127.0.0.1:8000/ready  

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness (models loaded) |
| `/analyze_message` | POST | Full analysis: spam, scam type, intel, risk score, recommended action |
| `/simulate_escalation` | POST | Rule-based scam escalation (offline, no LLM) |
| `/risk_report` | GET | Last analysis risk report |
| `/personas`, `/scam_types` | GET | Reference data |

### Analyze request/response

**POST /analyze_message**

```json
{
  "text": "URGENT! Your SBI account is blocked. Verify at http://sbi-fake.com. Call +91-9876543210",
  "sender": "+91-9876543210",
  "channel": "sms"
}
```

Response includes: `is_spam`, `spam_confidence`, `scam_type`, `scam_confidence`, `risk_score`, `risk_level`, `recommended_action`, `action_details`, `threat_intelligence` (urls, phones, money_amounts, organizations, threat_indicators), `features`, `processing_time_ms`.

### Simulate escalation (offline, rule-based)

**POST /simulate_escalation**

```json
{
  "scam_type": "bank_phishing",
  "num_turns": 5,
  "initial_message": "Your SBI account is blocked. Verify at http://sbi-fake.com"
}
```

Returns a list of escalation turns with `phase` and `scammer_message` (no external service calls).

## Docker

The Docker image includes:
- **Ollama** server for local LLM inference
- **Sarah model** (3B parameters) automatically downloaded from [Hugging Face](https://huggingface.co/Het0456/sarah)
- FastAPI server for spam/scam detection

### Build and Run

```bash
docker build -t spam-api .
docker run -p 8000:8000 -p 11434:11434 spam-api
```

**Note:** The first run will download the sarah.gguf model (~2-3GB), which may take several minutes. Subsequent runs will use the cached model.

### With docker-compose (Recommended)

```bash
docker-compose up --build
```

This will:
1. Start Ollama server on port 11434
2. Download and import the sarah model automatically
3. Start the FastAPI server on port 8000

**Memory Requirements:** The sarah model requires ~4-6GB RAM. Ensure Docker has at least 8GB allocated.

- API: http://localhost:8000
- Ollama: http://localhost:11434

## GCP deployment

**Note:** Cloud Run has memory limits and may not be suitable for running Ollama with the 3B sarah model. Consider using a GCP VM instead.

### Cloud Run (without Ollama)

For Cloud Run deployment, use `Dockerfile.cloudrun` which doesn't include Ollama:

```bash
docker build -f Dockerfile.cloudrun -t gcr.io/PROJECT_ID/spam-api .
docker push gcr.io/PROJECT_ID/spam-api
```

### GCP VM (with Ollama)

1. Build and push (replace `PROJECT_ID` and `REGION`):

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/spam-api
```

Or with Dockerfile.cloudrun (uses PORT=8080):

```bash
docker build -f Dockerfile.cloudrun -t gcr.io/PROJECT_ID/spam-api .
docker push gcr.io/PROJECT_ID/spam-api
```

2. Deploy:

```bash
gcloud run deploy spam-api \
  --image gcr.io/PROJECT_ID/spam-api \
  --platform managed \
  --region REGION \
  --allow-unauthenticated \
  --port 8080
```

Cloud Run sets `PORT` automatically. Use **/health** and **/ready** for health checks in the service configuration.

### GCP VM

1. Create a VM (e.g. e2-medium), SSH in.
2. Install Docker, clone repo, then:

```bash
cd spam_ai
docker build -t spam-api .
docker run -d -p 8000:8000 --restart unless-stopped --name spam-api spam-api
```

For more workers (higher throughput on a single VM):

```bash
docker run -d -p 8000:8000 -e WORKERS=4 --restart unless-stopped --name spam-api spam-api
```

(You would add a multi-worker CMD in the Dockerfile when `WORKERS` is set; current image uses a single uvicorn process. Scaling is typically done via multiple instances on Cloud Run or multiple VMs behind a load balancer.)

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `API_VERSION` | `1.0.0` | API version string |
| `MODEL_DIR` | `./models` | Path to model files |

## Project structure

```
spam_ai/
├── config.py           # Paths and env config
├── app/                # API layer
│   ├── main.py         # FastAPI app
│   ├── schemas.py      # Pydantic models
│   └── routes/         # Health, analyze, simulation, reports, chat, reference
├── core/               # Business logic
│   ├── model_loader.py # Load/cache spam & scam models
│   ├── risk_engine.py  # Risk score & recommendations
│   └── escalation_simulator.py  # Rule-based escalation (offline)
├── extraction/         # Text & threat intel
│   ├── preprocess.py
│   ├── features.py
│   └── extract_intel.py
├── bots/               # Conversational agents
│   └── scam_baiting_bot.py  # LLM-powered scam baiting
├── training/           # Model training scripts
│   ├── train_spam.py
│   └── train_scam_type.py
├── data/               # Training data
├── models/             # Trained model artifacts
├── Dockerfile
├── Dockerfile.cloudrun
├── docker-compose.yml
└── requirements.txt
```

## Ethical notice

For **educational and defensive use only**. No real attacker engagement; offline simulation only.
