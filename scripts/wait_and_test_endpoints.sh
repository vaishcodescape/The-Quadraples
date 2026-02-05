#!/usr/bin/env bash
# Wait for API and models to be ready, then test all endpoints.
set -e
BASE="${1:-http://localhost:8000}"
MAX_WAIT="${2:-300}"   # 5 minutes
INTERVAL=10

echo "=== Waiting for API and models to load (max ${MAX_WAIT}s) ==="
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
  if curl -sf "$BASE/ready" | grep -q '"ready":true'; then
    echo "✓ API and models are ready (after ${elapsed}s)"
    break
  fi
  echo "  Waiting... (${elapsed}s)"
  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
done

if [ "$elapsed" -ge "$MAX_WAIT" ]; then
  echo "✗ Timeout waiting for /ready"
  docker compose logs --tail 80 2>/dev/null || true
  exit 1
fi

echo ""
echo "=== Testing API endpoints ==="

# Health
echo -n "GET / ... "
curl -sf "$BASE/" | grep -q '"status":"healthy"' && echo "OK" || { echo "FAIL"; exit 1; }
echo -n "GET /health ... "
curl -sf "$BASE/health" | grep -q '"status":"ok"' && echo "OK" || { echo "FAIL"; exit 1; }
echo -n "GET /ready ... "
curl -sf "$BASE/ready" | grep -q '"ready":true' && echo "OK" || { echo "FAIL"; exit 1; }

# Reference
echo -n "GET /personas ... "
curl -sf "$BASE/personas" | grep -q '"personas"' && echo "OK" || { echo "FAIL"; exit 1; }
echo -n "GET /scam_types ... "
curl -sf "$BASE/scam_types" | grep -q '"scam_types"' && echo "OK" || { echo "FAIL"; exit 1; }

# Analyze
echo -n "POST /analyze_message ... "
code=$(curl -sf -o /tmp/analyze.json -w "%{http_code}" -X POST "$BASE/analyze_message" \
  -H "Content-Type: application/json" \
  -d '{"text":"URGENT! You won 1000000! Click http://fake.com now!","sender":"+91-9876543210","channel":"sms"}')
if [ "$code" = "200" ]; then
  grep -q '"risk_score"' /tmp/analyze.json && echo "OK" || { echo "FAIL (invalid response)"; exit 1; }
else
  echo "FAIL (HTTP $code)"; cat /tmp/analyze.json 2>/dev/null; exit 1
fi

# Risk report (after analyze)
echo -n "GET /risk_report ... "
curl -sf "$BASE/risk_report" | grep -q '"has_analysis"' && echo "OK" || { echo "FAIL"; exit 1; }

# Simulation (needs Ollama/sarah)
echo -n "POST /simulate_escalation ... "
code=$(curl -sf -o /tmp/sim.json -w "%{http_code}" -X POST "$BASE/simulate_escalation" \
  -H "Content-Type: application/json" \
  -d '{"scam_type":"bank_phishing","num_turns":2}')
if [ "$code" = "200" ]; then
  grep -q '"escalation"' /tmp/sim.json && echo "OK" || echo "WARN (check response)"
else
  echo "WARN (HTTP $code - Ollama may still be loading)"
fi

# Chat (needs Ollama/sarah)
echo -n "POST /chat/start ... "
code=$(curl -sf -o /tmp/chat_start.json -w "%{http_code}" -X POST "$BASE/chat/start" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-session","scammer_message":"Your bank account is blocked. Share OTP."}')
if [ "$code" = "200" ]; then echo "OK"; else echo "WARN (HTTP $code)"; fi

echo ""
echo "=== All critical endpoints OK ==="
