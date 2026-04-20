# Grimoire
Grimoire turns your documents into an AI that quizzes you on them, and gets harder as you get smarter. Upload course notes, textbooks, coding references, or that PDF you've been pretending to read. Grimoire ingests, chunks, and embeds the content, then generates quiz questions pulled directly from the source material. Get it right? The difficulty climbs. Get it wrong? That question is coming back later. Spaced repetition doesn't forget, even if you did. Under the hood: a retrieval-augmented generation pipeline with cross-encoder re-ranking for precision, an automated evaluation framework that measures retrieval quality and question relevance, and adaptive difficulty that tracks your performance per topic. Built with FastAPI, LangChain (LCEL), ChromaDB, and Gemini. Deployed on GCP Cloud Run with Prometheus/Grafana monitoring and CI/CD via GitHub Actions.

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

* Python 3.9 or later

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DevonV-byte/AutoQuizzer.git
   cd AutoQuizzer
   ```

2. **Create and activate a virtual environment:**
   * **On Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * **On macOS and Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Allowed document types
```bash
.docx
.pdf
.txt
.ipynb
.py
```
