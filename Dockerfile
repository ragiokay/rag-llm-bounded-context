FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY embedding/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model so runtime doesn't need internet
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source
COPY embedding/ ./embedding/

# Qdrant server URL — override at runtime with -e QDRANT_URL=...
ENV QDRANT_URL=http://localhost:6333

CMD ["python", "embedding/retrieve.py"]
