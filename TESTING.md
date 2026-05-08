# Ray AI Backend - Testing Guide

## Testing Strategy

Ray AI Backend menggunakan comprehensive testing approach:
- Health checks
- Unit tests (optional)
- Integration tests
- Manual API testing
- Error scenario testing

---

## 1. Health Check

### Basic Connectivity Test

```bash
# Should return 200 with status info
curl http://localhost:8000/

# Expected Response:
{
  "success": true,
  "status": "operational",
  "database": "healthy",
  "firebase": "enabled",
  "ai": "configured",
  "timestamp": "2026-04-25T10:00:00Z"
}
```

### Troubleshooting Health Check

**If database is "unhealthy":**
```bash
# Check PostgreSQL connection
psql -h localhost -U ray_user -d ray_ai_db -c "SELECT 1"

# Check DATABASE_URL in .env
cat .env | grep DATABASE_URL
```

**If firebase is "disabled":**
```bash
# Check firebase credentials file
ls -la firebase_credentials.json

# Verify file content is valid JSON
python -m json.tool firebase_credentials.json
```

**If ai is "not_configured":**
```bash
# Check GROQ_API_KEY in .env
cat .env | grep GROQ_API_KEY

# Verify API key is set
echo $GROQ_API_KEY
```

---

## 2. Authentication Testing

### Get Firebase Token (From Client)

```javascript
// In your frontend app
import { getAuth, signInWithPopup, GoogleAuthProvider } from "firebase/auth";

const auth = getAuth();
const provider = new GoogleAuthProvider();
const result = await signInWithPopup(auth, provider);
const token = await result.user.getIdToken();
console.log("Token:", token);
```

Or use Firebase CLI:

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Get token (interactive)
firebase auth:export tokens.json --project ray-ai-project
```

### Test Sync User

```bash
FIREBASE_TOKEN="your_token_here"

curl -X POST http://localhost:8000/auth/sync \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json"

# Expected Response:
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "firebase_uid": "abc123xyz",
    "email": "user@example.com",
    "name": "User Name",
    "profile_picture": "https://...",
    "created_at": "2026-04-25T10:00:00Z"
  },
  "timestamp": "2026-04-25T10:00:00Z"
}
```

### Error Testing - Auth

**Missing Authorization header:**
```bash
curl -X POST http://localhost:8000/auth/sync
# Expected: 401 - Missing Authorization header
```

**Invalid token format:**
```bash
curl -X POST http://localhost:8000/auth/sync \
  -H "Authorization: InvalidFormat"
# Expected: 401 - Invalid authorization header format
```

**Expired token:**
```bash
curl -X POST http://localhost:8000/auth/sync \
  -H "Authorization: Bearer expired_token_here"
# Expected: 401 - Token has expired
```

---

## 3. AI Endpoint Testing

### Test Learn Mode

```bash
FIREBASE_TOKEN="your_token_here"

curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa itu XSS dan bagaimana cara kerjanya?",
    "mode": "learn",
    "level": "beginner",
    "topic": "xss"
  }'

# Expected Response (will be lengthy):
{
  "success": true,
  "data": {
    "response": "XSS (Cross-Site Scripting) adalah...",
    "mode": "learn",
    "level": "beginner",
    "topic": "xss",
    "tokens_used": 245,
    "ai_success": true,
    "timestamp": "2026-04-25T10:00:00Z"
  },
  "timestamp": "2026-04-25T10:00:00Z"
}
```

### Test Chat Mode

```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa bedanya SQL injection dengan XSS?",
    "mode": "chat",
    "level": "intermediate"
  }'
```

### Test Practice Mode

```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Web Application Penetration Testing",
    "mode": "practice",
    "level": "intermediate",
    "topic": "web_security"
  }'
```

### Test Exploit Mode

```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "payload:'; DROP TABLE users;--",
    "mode": "exploit",
    "level": "advanced",
    "topic": "sqli"
  }'
```

### Error Testing - AI Endpoint

**Empty query:**
```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "", "mode": "learn", "level": "beginner"}'
# Expected: validation error
```

**Invalid mode:**
```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "mode": "invalid_mode", "level": "beginner"}'
# Expected: validation error
```

**Too long query (>2000 chars):**
```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "'$(python3 -c "print(\"x\" * 2001)")'"}'
# Expected: validation error
```

---

## 4. Progress Endpoint Testing

### Get User Progress

```bash
FIREBASE_TOKEN="your_token_here"

curl http://localhost:8000/progress \
  -H "Authorization: Bearer $FIREBASE_TOKEN"

# Expected Response:
{
  "success": true,
  "data": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "progress": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "user_id": "550e8400-e29b-41d4-a716-446655440000",
        "topic": "xss",
        "level": "beginner",
        "score": 85.5,
        "attempts": 3,
        "completed_count": 2,
        "last_attempted": "2026-04-25T09:00:00Z",
        "created_at": "2026-04-24T10:00:00Z"
      }
    ],
    "total_topics": 5
  },
  "timestamp": "2026-04-25T10:00:00Z"
}
```

### Update Progress

```bash
curl -X POST "http://localhost:8000/progress/update?topic=xss&level=beginner&score=92.5" \
  -H "Authorization: Bearer $FIREBASE_TOKEN"

# Expected Response:
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "topic": "xss",
    "level": "beginner",
    "score": 92.5,
    "attempts": 4,
    "completed_count": 3,
    "last_attempted": "2026-04-25T10:00:00Z",
    "created_at": "2026-04-24T10:00:00Z"
  },
  "timestamp": "2026-04-25T10:00:00Z"
}
```

### Error Testing - Progress

**Invalid score (>100):**
```bash
curl -X POST "http://localhost:8000/progress/update?topic=xss&level=beginner&score=150"
# Expected: error "Score must be between 0 and 100"
```

**Missing topic:**
```bash
curl -X POST "http://localhost:8000/progress/update?level=beginner&score=85"
# Expected: error "Missing required fields"
```

---

## 5. Knowledge Base Endpoint Testing

### Get Knowledge for Topic

```bash
FIREBASE_TOKEN="your_token_here"

curl http://localhost:8000/knowledge-base/xss \
  -H "Authorization: Bearer $FIREBASE_TOKEN"

# Expected Response:
{
  "success": true,
  "data": {
    "beginner": "XSS (Cross-Site Scripting) adalah serangan...",
    "intermediate": "XSS memiliki 3 tipe: Stored, Reflected, DOM-based...",
    "advanced": "Bypass techniques: encoding, HTML entities..."
  },
  "timestamp": "2026-04-25T10:00:00Z"
}
```

### Available Topics
- `xss` - Cross-Site Scripting
- `sqli` - SQL Injection
- `authentication` - Authentication attacks

---

## 6. Database Testing

### Connect to Database

```bash
# Using psql
psql -h localhost -U ray_user -d ray_ai_db

# Check tables
\dt

# Check users table
SELECT * FROM users;

# Check knowledge_outputs
SELECT * FROM knowledge_outputs LIMIT 5;

# Check user_progress
SELECT * FROM user_progress;
```

### Verify Data Integrity

```sql
-- Count users
SELECT COUNT(*) FROM users;

-- Count knowledge outputs
SELECT COUNT(*) FROM knowledge_outputs;

-- Check recent AI interactions
SELECT user_id, mode, level, created_at 
FROM knowledge_outputs 
ORDER BY created_at DESC 
LIMIT 10;

-- Check user progress
SELECT user_id, topic, level, score, attempts 
FROM user_progress 
ORDER BY created_at DESC 
LIMIT 10;

-- Find specific user
SELECT * FROM users WHERE email = 'user@example.com';
```

---

## 7. Error Scenario Testing

### Scenario 1: Database Connection Fails

**Setup:**
```bash
# Stop PostgreSQL
sudo systemctl stop postgresql

# Or if using Docker
docker stop ray_ai_db
```

**Expected Behavior:**
- Health check returns: `"database": "unhealthy"`
- `/auth/sync` returns 500 error
- `/ai` returns 500 error
- Errors logged to `logs/ray_ai.log`

**Verify:**
```bash
curl http://localhost:8000/
# Check "database": "unhealthy"

cat logs/ray_ai.log | tail -20
# Look for connection errors
```

**Recovery:**
```bash
# Restart PostgreSQL
sudo systemctl start postgresql

# Verify
curl http://localhost:8000/
# Should show "database": "healthy"
```

### Scenario 2: Firebase Credentials Missing

**Setup:**
```bash
mv firebase_credentials.json firebase_credentials.json.bak
```

**Expected Behavior:**
- Health check returns: `"firebase": "disabled"`
- `/auth/sync` returns 503 Service Unavailable
- `/ai` returns 401 Unauthorized
- Warning logged: "Firebase credentials not found"

**Recovery:**
```bash
mv firebase_credentials.json.bak firebase_credentials.json
# Restart server
```

### Scenario 3: Groq API Key Invalid

**Setup:**
```bash
# Edit .env
sed -i 's/GROQ_API_KEY=.*/GROQ_API_KEY=invalid_key/' .env
```

**Expected Behavior:**
- Health check returns: `"ai": "configured"` (key exists)
- `/ai` call returns fallback response
- Request is slow (due to retries)
- Error logged: "Groq API failed after all retries"

**Verify:**
```bash
# Check logs
tail -f logs/ray_ai.log

# Test AI endpoint
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
# Should return fallback response quickly
```

**Recovery:**
```bash
# Fix .env with correct key
# Restart server
```

### Scenario 4: Request Validation

**Invalid JSON:**
```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{invalid json}'
# Expected: 422 Unprocessable Entity
```

**Missing required field:**
```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mode": "learn"}'
# Expected: 422 - missing "query"
```

---

## 8. Performance Testing

### Latency Measurement

```bash
# Single request latency
time curl http://localhost:8000/

# AI request latency
FIREBASE_TOKEN="your_token_here"
time curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### Load Testing (with Apache Bench)

```bash
# Install ab
sudo apt install apache2-utils

# 100 concurrent requests
ab -n 100 -c 10 http://localhost:8000/
```

### Connection Pool Testing

```python
# Run in Python to test pool
import psycopg2
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://ray_user:ray_password@localhost:5432/ray_ai_db",
    poolclass=pool.QueuePool,
    pool_size=10,
    max_overflow=20,
)

# Create 30 connections
connections = []
for i in range(30):
    conn = engine.connect()
    connections.append(conn)
    print(f"Created connection {i+1}")

# Close all
for conn in connections:
    conn.close()
```

---

## 9. Automated Testing Script

### test_api.sh

```bash
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"
FIREBASE_TOKEN="your_token_here"

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function
test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    local expected_code=$5
    
    local cmd="curl -s -X $method $API_URL$endpoint"
    
    if [ ! -z "$data" ]; then
        cmd="$cmd -H 'Content-Type: application/json' -d '$data'"
    fi
    
    if [ ! -z "$FIREBASE_TOKEN" ]; then
        cmd="$cmd -H 'Authorization: Bearer $FIREBASE_TOKEN'"
    fi
    
    local response=$(eval $cmd)
    local http_code=$(eval "curl -s -o /dev/null -w '%{http_code}' $cmd")
    
    if [ "$http_code" == "$expected_code" ]; then
        echo -e "${GREEN}✓${NC} $name (HTTP $http_code)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗${NC} $name (Expected $expected_code, got $http_code)"
        ((TESTS_FAILED++))
    fi
}

# Run tests
echo "Starting Ray AI Backend Tests..."
echo ""

test_endpoint "Health Check" "GET" "/" "" "200"
test_endpoint "Missing Auth Header" "POST" "/auth/sync" "" "401"
test_endpoint "Invalid Token" "POST" "/auth/sync" "" "401"

# Add more tests...

echo ""
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
```

### Run tests:

```bash
chmod +x test_api.sh
./test_api.sh
```

---

## 10. Debugging Tips

### Enable Debug Logging

Edit `.env`:
```env
LOG_LEVEL=DEBUG
```

Or modify main.py:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Check Logs

```bash
# Real-time logs
tail -f logs/ray_ai.log

# Filter by level
grep "ERROR" logs/ray_ai.log
grep "WARNING" logs/ray_ai.log

# Search for specific request
grep "xss" logs/ray_ai.log
```

### Database Query Debugging

```bash
# Enable SQLAlchemy logging
# In main.py Config class:
echo "enable echo"  # For raw SQL logging
```

### API Documentation

Navigate to:
- `http://localhost:8000/docs` - Swagger UI
- `http://localhost:8000/redoc` - ReDoc

---

## Conclusion

Comprehensive testing ensures Ray AI Backend adalah:
- ✓ Reliable (error handling works)
- ✓ Performant (latency acceptable)
- ✓ Secure (auth verified)
- ✓ Scalable (pool management)

Run these tests regularly during development & before deployment!
