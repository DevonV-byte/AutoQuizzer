# Dockerfile for the Grimoire AutoQuizzer FastAPI backend.
# Packages the API server and its ChromaDB into a self-contained image.
# The GOOGLE_API_KEY must be supplied at runtime — it is never baked into the image.
#
# Build:
#   docker build -t autoquizzer .
#
# Run:
#   docker run -p 8000:8000 -e GOOGLE_API_KEY=<your_key> autoquizzer
#
# To use an external ChromaDB directory instead of the one baked into the image:
#   docker run -p 8000:8000 -e GOOGLE_API_KEY=<your_key> \
#       -v /path/to/Database:/app/Database autoquizzer

# --- Base image ---
FROM python:3.11-slim

# --- Working directory ---
WORKDIR /app

# --- Dependencies ---
# Copy requirements first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application source ---
COPY . .

# --- Runtime configuration ---
EXPOSE 8000

# FastAPI healthcheck — confirms the API is serving before Docker marks it healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --- Entrypoint ---
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
