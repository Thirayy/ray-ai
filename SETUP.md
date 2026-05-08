# Ray AI Backend - Setup & Deployment Guide

## Overview

Ray AI adalah backend production-ready untuk platform AI Cybersecurity Mentor. Dibangun dengan:
- **FastAPI** - Web framework modern
- **PostgreSQL** - Database relational
- **SQLAlchemy** - ORM
- **Firebase Admin SDK** - User authentication
- **Groq API** - LLM untuk AI responses

---

## Architecture

```
main.py (Single File)
├── 1. IMPORTS
├── 2. CONFIG (Settings)
├── 3. DATABASE SETUP (PostgreSQL + SQLAlchemy)
├── 4. FIREBASE SETUP (Auth)
├── 5. MODELS (User, KnowledgeOutput, UserProgress)
├── 6. SCHEMAS (Pydantic validation)
├── 7. UTILITY FUNCTIONS
├── 8. AUTH SYSTEM (Firebase verification)
├── 9. AI SYSTEM (RAG + Prompt Builder)
├── 10. CORE PIPELINE (Processing pipeline)
├── 11. ROUTES (FastAPI endpoints)
└── 12. ERROR HANDLERS
```

---

## Prerequisites

### System Requirements
- Python 3.9+
- PostgreSQL 13+
- Linux/Mac/Windows

### API Keys Required
1. **Firebase Service Account** - Get from Firebase Console
2. **Groq API Key** - Get from https://console.groq.com

---

## Installation

### 1. Clone & Setup

```bash
cd /home/zar/ray-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

#### Option A: Using Docker (Recommended)

```bash
# Run PostgreSQL with Docker
docker run -d \
  --name ray_ai_db \
  -e POSTGRES_USER=ray_user \
  -e POSTGRES_PASSWORD=ray_password \
  -e POSTGRES_DB=ray_ai_db \
  -p 5432:5432 \
  postgres:15

# Verify connection
psql -h localhost -U ray_user -d ray_ai_db
```

#### Option B: Manual PostgreSQL Installation

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib  # Ubuntu/Debian
brew install postgresql                         # macOS

# Create database
createdb -U postgres ray_ai_db
createuser -U postgres ray_user --createdb
psql -U postgres -c "ALTER USER ray_user WITH PASSWORD 'ray_password';"
```

### 3. Environment Setup

Create `.env` file:

```env
# Database
DATABASE_URL=postgresql://ray_user:ray_password@localhost:5432/ray_ai_db

# Firebase
FIREBASE_CREDENTIALS_PATH=firebase_credentials.json

# Groq API
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Firebase Setup

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create new project
3. Go to Settings → Service Accounts
4. Click "Generate New Private Key"
5. Save as `firebase_credentials.json` in project root

### 5. Run the Server

```bash
# Development
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production (with Gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

The API will be available at `http://localhost:8000`

---

## API Endpoints

### Health Check

```bash
GET /
```

Response:
```json
{
  "success": true,
  "status": "operational",
  "database": "healthy",
  "firebase": "enabled",
  "ai": "configured",
  "timestamp": "2026-04-25T10:00:00Z"
}
```

### Auth - Sync User

```bash
POST /auth/sync
Authorization: Bearer <firebase_token>
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "firebase_uid": "uid",
    "email": "user@example.com",
    "name": "User Name",
    "profile_picture": "url",
    "created_at": "2026-04-25T10:00:00Z"
  },
  "timestamp": "2026-04-25T10:00:00Z"
}
```

### AI - Ask Question

```bash
POST /ai
Authorization: Bearer <firebase_token>
Content-Type: application/json

{
  "query": "Apa itu XSS?",
  "mode": "learn",
  "level": "beginner",
  "topic": "xss"
}
```

Modes: `chat`, `learn`, `practice`, `exploit`
Levels: `beginner`, `intermediate`, `advanced`

Response:
```json
{
  "success": true,
  "data": {
    "response": "XSS adalah...",
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

### Progress - Get User Progress

```bash
GET /progress
Authorization: Bearer <firebase_token>
```

Response:
```json
{
  "success": true,
  "data": {
    "user_id": "uuid",
    "progress": [
      {
        "id": "uuid",
        "topic": "xss",
        "level": "beginner",
        "score": 85.5,
        "attempts": 3,
        "completed_count": 2
      }
    ],
    "total_topics": 5
  }
}
```

### Progress - Update Progress

```bash
POST /progress/update?topic=xss&level=beginner&score=85.5
Authorization: Bearer <firebase_token>
```

---

## Database Schema

### users
```
- id (UUID) - Primary key
- firebase_uid (String) - Firebase UID, unique
- email (String) - User email, unique
- name (String) - User name
- profile_picture (String) - Profile image URL
- created_at (DateTime) - Created timestamp
- updated_at (DateTime) - Updated timestamp
```

### knowledge_outputs
```
- id (UUID) - Primary key
- user_id (UUID) - Foreign key to users
- mode (String) - chat/learn/practice/exploit
- level (String) - beginner/intermediate/advanced
- topic (String) - Topic name
- query (Text) - User query
- content (JSONB) - Response content
- tokens_used (Integer) - LLM tokens used
- created_at (DateTime) - Created timestamp
```

### user_progress
```
- id (UUID) - Primary key
- user_id (UUID) - Foreign key to users
- topic (String) - Topic name
- level (String) - beginner/intermediate/advanced
- score (Float) - Score 0-100
- attempts (Integer) - Number of attempts
- completed_count (Integer) - Completed count
- last_attempted (DateTime) - Last attempt time
- created_at (DateTime) - Created timestamp
```

---

## Error Handling

System handles:

### Authentication Errors
- Missing Authorization header → 401
- Invalid token format → 401
- Expired token → 401
- Firebase verification failed → 401

### Database Errors
- Connection failed → logged, fallback response
- Query error → rolled back, error response
- Transaction failed → rolled back, 500

### AI Errors
- API timeout → retry with exponential backoff
- Rate limited (429) → retry
- API failure → fallback to knowledge base
- Timeout on all retries → fallback response

### Validation Errors
- Empty query → 400
- Invalid mode/level → 400
- Invalid score → 400

---

## Failsafe System

### If Database Fails
- Error logged to file
- Response still returned from AI
- Knowledge output not saved but user gets response

### If AI Fails
- Retries 3 times with exponential backoff
- If all retries fail, returns fallback response from knowledge base
- System never crashes

### If Firebase Fails
- Auth endpoint returns 503
- AI endpoint returns 401
- System gracefully handles disabled auth

---

## Configuration

### Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Firebase
FIREBASE_CREDENTIALS_PATH=firebase_credentials.json

# Groq AI
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TIMEOUT=30
GROQ_MAX_RETRIES=3

# API
API_HOST=0.0.0.0
API_PORT=8000

# CORS
ALLOWED_ORIGINS=http://localhost:3000,https://ray-ai.com
```

### Database Connection Pool

```python
pool_size=10         # Base pool size
max_overflow=20      # Additional connections allowed
pool_pre_ping=True   # Check connection before using
```

---

## Monitoring & Logging

### Log Levels
- **INFO**: User actions, system events
- **WARNING**: Fallbacks, retries
- **ERROR**: Failures, exceptions

### Log File
```
logs/ray_ai.log
```

### Key Events Logged
- User created/updated
- Knowledge output saved
- AI API calls (success/failure)
- Database operations
- Auth verification
- Errors and exceptions

---

## Testing

### Health Check Test

```bash
curl http://localhost:8000/
```

### Auth Test

```bash
# Get Firebase token first (from client)
FIREBASE_TOKEN="your_token_here"

curl -X POST http://localhost:8000/auth/sync \
  -H "Authorization: Bearer $FIREBASE_TOKEN"
```

### AI Endpoint Test

```bash
curl -X POST http://localhost:8000/ai \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa itu XSS?",
    "mode": "learn",
    "level": "beginner",
    "topic": "xss"
  }'
```

---

## Production Deployment

### 1. Use Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

### 2. Use Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY firebase_credentials.json .
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t ray-ai-backend .
docker run -p 8000:8000 -e DATABASE_URL=postgresql://... ray-ai-backend
```

### 3. Use Reverse Proxy (Nginx)

```nginx
upstream ray_ai {
    server localhost:8000;
}

server {
    listen 80;
    server_name api.ray-ai.com;

    location / {
        proxy_pass http://ray_ai;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. SSL/TLS Setup

```bash
# Using certbot with Letsencrypt
sudo certbot certonly --standalone -d api.ray-ai.com
```

---

## Troubleshooting

### Database Connection Error

```
Error: could not connect to server
```

Solution:
- Check PostgreSQL is running: `psql -U postgres`
- Verify DATABASE_URL in .env
- Check firewall rules

### Firebase Error

```
Error: invalid_grant
```

Solution:
- Check firebase_credentials.json is valid
- Verify it's in the correct path
- Check credentials have right permissions

### AI Timeout

```
Error: Groq API timeout
```

Solution:
- Check GROQ_API_KEY is valid
- Verify internet connection
- Increase GROQ_TIMEOUT in .env

### Port Already in Use

```
Error: Address already in use
```

Solution:
```bash
# Find process using port 8000
lsof -i :8000
# Kill process
kill -9 <PID>
```

---

## API Documentation

Interactive docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Security Considerations

1. **Firebase Tokens**
   - Always validate on backend
   - Set expiration times
   - Rotate credentials periodically

2. **Database**
   - Use strong passwords
   - Enable SSL connections
   - Regular backups

3. **API**
   - Use HTTPS in production
   - Implement rate limiting
   - Add CORS restrictions

4. **Secrets**
   - Never commit `.env` file
   - Use environment variables in production
   - Rotate API keys regularly

---

## Performance Tuning

### Database Optimization
- Create indexes on frequently queried columns
- Use connection pooling
- Enable query caching

### API Optimization
- Cache knowledge base responses
- Implement pagination for large result sets
- Use gzip compression

### AI Optimization
- Batch similar requests
- Cache prompt templates
- Use async/await for long operations

---

## Support

For issues or questions:
1. Check logs: `logs/ray_ai.log`
2. Review troubleshooting section
3. Check FastAPI docs: https://fastapi.tiangolo.com/

---

## Version History

- **v1.0.0** (2026-04-25) - Initial release
  - Authentication with Firebase
  - AI learning system
  - Progress tracking
  - RAG integration
  - Comprehensive error handling
