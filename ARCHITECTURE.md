# Ray AI Backend - Architecture Documentation

## System Overview

Ray AI adalah backend sistem untuk platform AI Cybersecurity Mentor yang dibangun dengan prinsip:
- **Single File Architecture** - Mudah di-deploy dan maintain
- **Production-Ready** - Error handling, logging, monitoring
- **Scalable Design** - Connection pooling, async operations
- **Defensive Programming** - Failsafe system, graceful degradation

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│              (Web/Mobile App dengan Firebase SDK)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    Firebase Token (JWT)
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      API LAYER (FastAPI)                        │
├──────────────────────────────────────────────────────────────────┤
│  POST /auth/sync      - Sync user ke database                    │
│  POST /ai             - Main AI endpoint                         │
│  GET  /progress       - Get user progress                        │
│  POST /progress/update - Update progress                         │
│  GET  /knowledge-base/{topic} - Get knowledge                    │
│  GET  /              - Health check                              │
└──────────┬────────────────┬────────────────┬────────────────────┘
           │                │                │
    ┌──────▼─────┐   ┌─────▼──────┐   ┌────▼────────┐
    │   AUTH     │   │     AI     │   │  DATABASE   │
    │  LAYER     │   │   SYSTEM   │   │   LAYER     │
    └────────────┘   └────────────┘   └─────────────┘
           │                │                │
      Firebase       Groq API + RAG    PostgreSQL
                   + Fallback KB      + SQLAlchemy
```

---

## Component Architecture (Dalam main.py)

### 1. IMPORTS (Lines 1-40)
Mengimport semua dependencies:
- FastAPI, Uvicorn, Pydantic
- SQLAlchemy ORM + PostgreSQL
- Firebase Admin SDK
- Requests untuk HTTP calls
- Logging, JSON, typing

### 2. CONFIG (Lines 42-65)
Global settings configuration:
- Database URL
- Firebase credentials path
- Groq API key
- API host/port
- CORS origins

### 3. DATABASE SETUP (Lines 67-110)
Database initialization dan session management:
- SQLAlchemy engine dengan connection pooling
- SessionLocal untuk dependency injection
- `get_db()` dependency untuk automatic session management
- `init_db()` untuk create tables on startup

```python
# Connection Pool Configuration
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=pool.QueuePool,
    pool_size=10,              # Base connections
    max_overflow=20,           # Additional connections
    pool_pre_ping=True,        # Health check before use
)
```

### 4. FIREBASE SETUP (Lines 112-135)
Firebase Admin SDK initialization:
- Loads credentials dari JSON file
- Initialize Firebase app
- Error handling jika credentials missing

### 5. MODELS (Lines 137-230)
SQLAlchemy ORM models untuk database:

**User Model**
```
- id (UUID, PK)
- firebase_uid (unique)
- email (unique)
- name
- profile_picture
- created_at, updated_at
```

**KnowledgeOutput Model**
```
- id (UUID, PK)
- user_id (FK → users)
- mode (chat/learn/practice/exploit)
- level (beginner/intermediate/advanced)
- topic
- query (user's input)
- content (JSON response)
- tokens_used
- created_at
```

**UserProgress Model**
```
- id (UUID, PK)
- user_id (FK → users)
- topic
- level
- score (0-100)
- attempts
- completed_count
- last_attempted
- created_at
```

### 6. SCHEMAS (Lines 232-310)
Pydantic models untuk request/response validation:

```python
# Enums
AIMode: chat, learn, practice, exploit
Level: beginner, intermediate, advanced

# Request Schema
AIRequest:
  - query (required, 1-2000 chars)
  - mode (default: learn)
  - level (default: beginner)
  - topic (optional)

# Response Schema
AIResponse:
  - success (bool)
  - data (dict)
  - error (str)
  - timestamp (ISO format)
```

### 7. UTILITY FUNCTIONS (Lines 312-400)
Helper functions:

- `safe_parse_json()` - Safe JSON parsing
- `create_error_response()` - Standardized error responses
- `create_success_response()` - Standardized success responses
- `KNOWLEDGE_BASE` - Fallback knowledge untuk berbagai topics

### 8. AUTH SYSTEM (Lines 402-520)

**`verify_firebase_token()`**
Flow:
1. Read "Authorization" header
2. Validate "Bearer <token>" format
3. Call Firebase `auth.verify_id_token()`
4. Extract: uid, email, name, picture
5. Return decoded token atau raise 401

Error handling:
- ExpiredIdTokenError → 401
- InvalidIdTokenError → 401
- Missing header → 401
- Invalid format → 401

**`sync_or_create_user()`**
Flow:
1. Get uid dari Firebase token
2. Query database untuk existing user
3. Jika ada: update name, picture, updated_at
4. Jika tidak: create new user
5. Return user object

Error handling:
- SQLAlchemy errors → rollback + 500
- All exceptions caught dengan try/except

### 9. AI SYSTEM (Lines 522-680)

**AISystem class** - Menangani AI response generation:

```python
class AISystem:
    ├── build_context()      # Build RAG context
    ├── build_prompt()       # Build LLM prompt
    ├── call_groq_api()      # Call Groq dengan retries
    └── generate_response()  # Main AI generation
```

**`build_context()`**
Membuat konteks dari:
- Topic-specific knowledge base entries
- User query
- Ragmentasi knowledge base

**`build_prompt()`**
Membangun prompt berbeda untuk setiap mode:

```
LEARN mode → Structured learning (Intuisi, Konsep, Workflow, etc)
CHAT mode → Conversational answer
PRACTICE mode → Challenge creation
EXPLOIT mode → Exploitation analysis
```

**`call_groq_api()`**
Retry logic dengan exponential backoff:

```python
For attempt in range(max_retries):
    Try:
        POST to Groq API with timeout
    If 200:
        Return success + content + tokens
    If 429 (Rate Limited):
        Wait 2^attempt seconds
        Retry
    If timeout/error:
        Log error
        Retry
If all retries fail:
    Return False (trigger fallback)
```

**`generate_response()`**
Main flow:
1. Build context
2. Build prompt
3. Call Groq API
4. If success → return response
5. If fail → return fallback dari knowledge base

### 10. CORE PIPELINE (Lines 682-790)

**ResponsePipeline class** - Main processing pipeline:

```python
class ResponsePipeline:
    @staticmethod
    def process_ai_request():
        Flow:
        1. Validate request (query tidak kosong)
        2. Get user (dari auth layer)
        3. Generate AI response
        4. Save ke database (dengan try/except)
        5. Return response
        
        Design: Jika DB save fails, response tetap dikembalikan
```

### 11. ROUTES (Lines 792-1100)

**FastAPI routes implementation:**

```
GET  /
    ├── Health check
    └── Returns: database, firebase, ai status

POST /auth/sync
    ├── Verify Firebase token
    ├── Sync user ke DB
    └── Returns: User object

POST /ai
    ├── Verify Firebase token
    ├── Process request through pipeline
    └── Returns: AI response + metadata

GET  /progress
    ├── Verify Firebase token
    ├── Query user_progress table
    └── Returns: User's progress list

POST /progress/update
    ├── Verify Firebase token
    ├── Create/update progress record
    └── Returns: Updated progress

GET  /knowledge-base/{topic}
    ├── Verify Firebase token
    ├── Return knowledge for topic
    └── Returns: Knowledge by level
```

### 12. ERROR HANDLERS (Lines 1102-1140)

**Global exception handlers:**

```python
@app.exception_handler(HTTPException)
    ├── Log error
    └── Return structured error response

@app.exception_handler(Exception)
    ├── Log full traceback
    └── Return 500 with generic message
```

---

## Data Flow Diagram

### Request Flow (AI Endpoint)

```
CLIENT REQUEST
      ↓
  /ai endpoint
      ↓
verify_firebase_token()
  ├─ Check Authorization header
  ├─ Verify token with Firebase
  ├─ Extract uid, email, name
  └─ Return decoded token
      ↓
sync_or_create_user()
  ├─ Check if user exists in DB
  ├─ Create if not exists
  └─ Return User object
      ↓
ResponsePipeline.process_ai_request()
  ├─ Validate request
  ├─ Generate AI response
  │   ├─ build_context()
  │   ├─ build_prompt()
  │   ├─ call_groq_api()
  │   └─ Fallback to knowledge base
  │
  └─ Save to database (try/except)
      ├─ Create KnowledgeOutput
      ├─ db.add() & db.commit()
      └─ If fails: log & continue
      
      ↓
  RETURN RESPONSE
```

### Error Handling Flow

```
TRY AI CALL
    ↓
[Success] → Return response
    ↓
[Timeout] → Retry with backoff
    ↓
[Rate Limited (429)] → Retry
    ↓
[API Error] → Continue retrying
    ↓
[All Retries Failed] → Use Fallback
    ↓
Return fallback response from knowledge base
```

---

## Database Schema

### Entity Relationship Diagram

```
┌──────────────┐
│    users     │
├──────────────┤
│ id (PK)      │
│ firebase_uid │
│ email        │
│ name         │
│ picture      │
│ created_at   │
│ updated_at   │
└──────┬───────┘
       │
       │ 1:N
       │
    ┌──┴─────────────────┐
    │                    │
    ↓                    ↓
┌──────────────┐   ┌──────────────────┐
│knowledge_    │   │ user_progress    │
│outputs       │   ├──────────────────┤
├──────────────┤   │ id (PK)          │
│ id (PK)      │   │ user_id (FK)     │
│ user_id (FK) │   │ topic            │
│ mode         │   │ level            │
│ level        │   │ score            │
│ topic        │   │ attempts         │
│ query        │   │ completed_count  │
│ content      │   │ last_attempted   │
│ tokens_used  │   │ created_at       │
│ created_at   │   └──────────────────┘
└──────────────┘
```

---

## Authentication Flow

```
CLIENT (Mobile/Web App)
    ↓
User clicks "Login dengan Google"
    ↓
Firebase SDK (client-side)
    ├─ Redirect ke Google OAuth
    ├─ User grants permission
    └─ Receive id_token (JWT)
    ↓
Store token in localStorage/secure storage
    ↓
API REQUEST
    ↓
Set header: "Authorization: Bearer <token>"
    ↓
SERVER receives request
    ↓
verify_firebase_token()
    ├─ Extract token dari header
    ├─ Call Firebase.auth.verify_id_token(token)
    ├─ Verify signature & expiration
    └─ Return decoded claims
    ↓
sync_or_create_user()
    ├─ Check if user exists by firebase_uid
    ├─ Create or update user in DB
    └─ Return user object
    ↓
Process request dengan user context
    ↓
Return response
```

---

## AI Response Generation Flow

```
USER QUERY: "Apa itu XSS?"
    ↓
AIRequest(
  query="Apa itu XSS?",
  mode="learn",
  level="beginner",
  topic="xss"
)
    ↓
AISystem.generate_response()
    │
    ├─ Step 1: build_context()
    │   └─ Context = topic knowledge + query
    │
    ├─ Step 2: build_prompt()
    │   ├─ For mode=learn:
    │   │   "You are Ray mentor. Provide structured learning response..."
    │   └─ Prompt includes: mode, level, query, context
    │
    ├─ Step 3: call_groq_api()
    │   ├─ POST to https://api.groq.com/openai/v1/chat/completions
    │   ├─ With model: "llama-3.3-70b-versatile"
    │   │
    │   ├─ Retry logic:
    │   │   Attempt 1 → (timeout) → Wait 1s → Retry
    │   │   Attempt 2 → (429) → Wait 2s → Retry
    │   │   Attempt 3 → (success) → Return content
    │   │
    │   └─ Extract: response, tokens_used
    │
    └─ If all fail:
        └─ fallback_response = KNOWLEDGE_BASE["xss"]["beginner"]
    ↓
KnowledgeOutput.create()
    ├─ user_id, mode, level, query
    ├─ content, tokens_used
    └─ db.commit()
    ↓
RESPONSE:
{
  "success": true,
  "data": {
    "response": "XSS adalah serangan...",
    "mode": "learn",
    "level": "beginner",
    "topic": "xss",
    "tokens_used": 245,
    "ai_success": true,
    "timestamp": "2026-04-25T10:00:00Z"
  }
}
```

---

## Failsafe System Design

### Scenario 1: Groq API Fails

```
AISystem.call_groq_api() → All retries exhausted
    ↓
generate_response() returns: (False, fallback_response, 0)
    ↓
Return response dengan fallback content
    ↓
User tetap dapat jawaban berkualitas dari knowledge base
    ↓
Status logged untuk monitoring
```

### Scenario 2: Database Fails

```
KnowledgeOutput creation → SQLAlchemy error
    ↓
Caught in ResponsePipeline.process_ai_request()
    ↓
db.rollback()
    ↓
Log error
    ↓
Return response to user ANYWAY
    ├─ User gets AI response
    ├─ Knowledge output not saved
    └─ But conversation still works
```

### Scenario 3: Firebase Fails

```
Firebase Admin SDK not initialized
    ↓
FIREBASE_ENABLED = False
    ↓
/auth/sync → 503 Service Unavailable
    ↓
/ai → 401 Unauthorized (Firebase verification failed)
    ↓
System continues operational without auth
    ├─ Health check shows: "firebase": "disabled"
    └─ Admin can handle manually
```

---

## Security Architecture

### Authentication Layer
- Firebase tokens validated server-side
- JWT signature verification
- Token expiration checks
- Per-request authentication

### Database Security
- SQLAlchemy prevents SQL injection (parameterized queries)
- Connection pooling dengan SSL support
- User data isolated by firebase_uid
- No sensitive data in logs

### API Security
- CORS validation untuk localhost & ray-ai.com
- Authorization header required
- Request validation dengan Pydantic
- Error messages don't expose internals

---

## Performance Characteristics

### Database
- Connection pool: 10 base + 20 overflow = 30 concurrent
- Query indices: user_id, firebase_uid, email
- Async SQLAlchemy support ready

### API
- FastAPI async/await ready
- Uvicorn dengan multiple workers
- Request timeout: 30s (Groq API)
- Fallback response time: <100ms

### AI System
- Groq API latency: ~2-5s typical
- Retry backoff: exponential (1s, 2s, 4s...)
- Fallback response: immediate
- Token tracking untuk billing

---

## Monitoring & Observability

### Logging Strategy
```
INFO  - User actions (auth, queries)
        AI calls (success/failure)
        Database operations

WARNING - API retries
          Fallback usage
          DB connection issues

ERROR - Auth failures
        API permanent failures
        DB transaction rollbacks
```

### Health Check Endpoint
```
GET / → Returns:
  - database: healthy/unhealthy
  - firebase: enabled/disabled/error
  - ai: configured/not_configured
```

### Metrics to Track
- Response latency (p50, p95, p99)
- Error rates
- AI fallback rate
- DB connection pool usage
- Token consumption (Groq)

---

## Deployment Architecture

### Development
```
localhost:8000 ← uvicorn --reload
├─ PostgreSQL (Docker atau local)
├─ Firebase (test project)
└─ Groq API (sandbox)
```

### Production
```
                    Client
                      ↓
            [HTTPS / CloudFlare]
                      ↓
        [Nginx Reverse Proxy]
                      ↓
    [Gunicorn - 4 workers]
     ├─ main.py instance 1
     ├─ main.py instance 2
     ├─ main.py instance 3
     └─ main.py instance 4
                      ↓
        [Database Connection Pool]
                      ↓
    [PostgreSQL RDS]
```

---

## Scalability Considerations

### Horizontal Scaling
- Stateless API design (easy to scale)
- Database connection pooling
- Session per request (no session state)

### Vertical Scaling
- Worker processes (Gunicorn -w flag)
- Connection pool size (adjustable)
- Memory per instance

### Future Optimizations
- Redis caching layer
- Message queue (Celery)
- Vector database (pgvector)
- Rate limiting (Redis)
- Request batching

---

## Conclusion

Ray AI backend adalah production-ready system yang:
- Mudah di-maintain (single file)
- Reliable (comprehensive error handling)
- Secure (Firebase auth, DB protection)
- Scalable (pool, async ready)
- Observable (logging, health checks)

Didesain untuk handle real-world scenarios dengan graceful degradation.
