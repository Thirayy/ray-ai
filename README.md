
# Ray AI

**Offline-First Cyber Security Research Engine** for education, laboratories, and technical demonstrations.

Ray AI is a specialized AI system focused on cyber security research, defensive analysis, and transparent knowledge retrieval. It is designed to operate both **online** and **offline**, making it suitable for classrooms, exhibitions, controlled lab environments, and low-connectivity deployments.

---

## 🚀 Why Ray AI?

Most AI assistants are general-purpose chat systems. Ray AI takes a different approach:

- Focused specifically on **cyber security**
- Prioritizes **technical analysis over casual conversation**
- Supports **offline operation**
- Uses **auditable local knowledge sources**
- Generates structured outputs for learning and research
- Designed for demonstrations, schools, and lab use

---

## 🎯 Core Use Cases

- Cyber security education
- Vulnerability awareness training
- Defensive security research
- Technical exhibitions / science fairs
- Offline AI demos
- Internal lab environments

---

## ✨ Key Features

### 🔐 Cyber Security Focused
Ray AI is built for security topics such as:

- Web vulnerabilities
- Network defense concepts
- Risk analysis
- Security frameworks
- Incident response concepts
- OWASP / best practices
- Threat awareness

### 🌐 Dual Operation Modes

#### Online Mode
Uses external LLM APIs for fast and dynamic responses.

- Low latency
- Strong reasoning
- Ideal for live demos

#### Offline Mode
Uses local datasets and vector search.

- No internet required
- Stable in closed environments
- Transparent sources
- Great for schools & exhibitions

#### Hybrid Mode
Automatically switches between online and offline sources when needed.

---

## 🧠 Output Style

Ray AI is optimized for direct, structured, technical responses.

Example output:

```text
Query:
Analyze brute-force risk on SSH exposed to the internet

Response:
Risk Level: Medium
Likelihood: High

Recommended Controls:
- Disable password authentication
- Use SSH keys only
- Enable Fail2Ban
- Restrict IP access
- Monitor login attempts
````

---

## 🧱 Architecture

```text
User Query
   │
   ▼
Context Router
   │
   ├── Online LLM API
   │
   └── Offline Vector Knowledge Base
   │
   ▼
Response Engine
   │
   ▼
Structured Technical Output
```

---

## 🛠️ Tech Stack

* Python
* Vector Search / FAISS
* Local PDF Knowledge Base
* HTML / CSS Frontend
* Optional External LLM API

---

## 📁 Project Structure

```text
ray-ai/
├── ray_ai.py
├── build_kb.py
├── kb/
├── static/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Thirayy/ray-ai.git
cd ray-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Build Knowledge Base

```bash
python build_kb.py
```

### 4. Run Application

```bash
python ray_ai.py
```

---

## 📚 Knowledge Sources

Ray AI can use:

* Local PDFs
* Security documentation
* Whitepapers
* Internal datasets
* Optional external AI APIs

All sources are replaceable and auditable.

---

## 🧪 Designed for Safe Environments

Ray AI is intended for:

* Education
* Defensive learning
* Technical research
* Controlled demonstrations

It is not intended for unauthorized activity.

---

## 🎪 Exhibition / Judge Friendly

Ray AI is especially useful for competitions and public demos because it offers:

* Offline capability
* Clear technical focus
* Transparent knowledge base
* Strong educational value
* Practical AI implementation

---

## 🗺️ Roadmap

Planned future improvements:

* Web dashboard
* Multi-language support
* Better document ingestion
* SOC-style reporting mode
* Voice interaction
* Local model integration

---

## 💬 Tagline

**Ray AI — Built for the Lab, Not for Small Talk**

---

## 📄 License

This project is intended for educational and research purposes.

```
```
