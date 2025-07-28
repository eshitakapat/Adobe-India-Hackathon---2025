FROM python:3.10-slim

ENV TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache

WORKDIR /app

COPY . /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pdfplumber \
    torch \
    sentence-transformers

# Preload model to cache (optional, speeds up runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

CMD ["python", "process_pdfs.py"]