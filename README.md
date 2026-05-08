# Ray AI — Cyber Security Research Engine

## 📌 Project Identity

**Name:** Ray AI  
**Category:** AI Cyber Security Research Engine  
**Platform:** Linux  
**Operation Mode:** Online & Offline  
**Target:** Education, cyber security research, technology exhibition

## 🎯 Main Purpose

- Provide AI specialized in cyber security
- Deliver raw technical analysis output
- No conversational chatbot safety layer
- Usable without internet
- Suitable for lab environments, schools, and public demos

## ✨ Key Differentiators from ChatGPT & Other AIs

- Not a general chatbot
- Research-oriented, not conversational
- Does not polish answers to be "comfortable"
- Focus solely on cyber security
- Transparent data sources (PDF / local dataset)
- Offline-first capability
- Output in technical analysis format, not opinions

## 🧠 Output Philosophy (Important)

### Ray AI DOES:
- Explain technical mechanisms
- Analyze vulnerabilities conceptually
- Explain mitigation & defense strategies
- Present analysis like SOC reports / whitepapers

### Ray AI DOES NOT:
- Roleplay as hackers
- Provide real-world exploit instructions
- Perform automation attacks

*This creates difference, not lack of ethics.*

## 🔧 Operation Modes

### 🔹 1. Online Mode
- Uses Groq API
- Very fast response
- Suitable for live demos
- Used when internet is available

> ⚠️ Groq API limits
> - Groq enforces rate limits and quotas that cannot be bypassed by the client.
- Ray AI implements client-side throttling, exponential backoff (with Retry-After support), a circuit-breaker and a deterministic offline fallback so demos continue to work when the API is rate-limited.

> Metrics: Ray AI collects simple, local metrics for Groq calls (success/failure/retries/rate-limited) so you can observe API health during demos.

### 🔹 2. Offline Mode
- Uses PDF Knowledge Base
- FAISS vector search
- Works without internet
- Suitable for closed labs & exhibitions

### 🔹 3. Hybrid Mode
- Auto switch online ↔ offline
- Online = Groq
- Offline = FAISS PDF

## 📚 Dataset & Knowledge Sources

- API Groq (LLM)
- Local PDFs (whitepapers, OWASP, security docs)
- Static datasets (offline)

*All data: auditable, replaceable, not black-box*

## ⚙️ System Architecture

```
User Query
   │
   ▼
Context Engine
   │
   ├── Online → Groq API
   │
   └── Offline → FAISS PDF Index
   │
   ▼
Ray AI Response Engine
   │
   ▼
Raw Technical Output + Source Reference
```

## 📁 Project Structure

```
ray-ai/
├── ray_ai.py              # Main application
├── build_kb.py            # Knowledge base builder
├── kb/                    # Knowledge base directory
│   ├── faiss.index        # Vector index
│   └── metadata.pkl       # Metadata storage
├── static/index.html      # Frontend interface
├── requirements.txt       # Dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## 🔐 Security & Academic Focus

- Not used for real attacks
- Focus on education & research
- Controlled environment
- Suitable for school & exhibition contexts

## 🎪 Demo Mode (For Exhibitions)

- Static dataset
- Safe questions
- Technical output
- No dangerous input
- Not connected to real targets

## 💎 Value Proposition for Teachers & Judges

- Offline AI (rare)
- Cyber security focus
- Source transparency
- Not a general chatbot
- Suitable for education

## 🗣️ Key Presentation Sentence

*"Ray AI is a cyber security research AI that displays raw technical analysis from transparent data sources, and can operate offline."*

## 🏷️ Tagline Options

- Ray AI — Cyber Security, Without the Chat Layer
- Ray AI — Research First, Chat Later
- Ray AI — Built for the Lab, Not for Small Talk

## 🛠️ Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build the knowledge base:
```bash
python build_kb.py
```

3. Run the application:
```bash
python ray_ai.py
```

4. Presentation/demo (offline sample)
```bash
# renders a polished demo report (no network)
./scripts/demo_presentation.sh
```
This creates `static/demo/demo-report.md` and `static/demo/demo-report.html` suitable for slides or judge review.

### Pretty output (terminal + markdown)
You can produce AI/Claude‑style boxed terminal summaries or slide‑ready Markdown using the pretty-printer:

```bash
# terminal box demo
python3 -c "from tools.pretty_output import render_box; print(render_box(['Hosts: 1','Open ports: 2'], title='Scan summary'))"

# SQLi slide snippet (Markdown)
python3 -c "from tools.pretty_output import vuln_explanation_sql_injection; print(vuln_explanation_sql_injection())"
```


## 📄 License

This project is intended for educational purposes in cyber security research.