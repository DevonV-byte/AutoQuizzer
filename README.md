# Grimoire
An interactive quiz application demonstrating Retrieval-Augmented Generation (RAG). Built using LangChain Expression Language (LCEL), Google's Gemini LLM, and ChromaDB for local vector embeddings, all wrapped in a Streamlit interface.

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