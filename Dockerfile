FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy helper scripts (imported by the webapp)
COPY scripts/ ./scripts/

# Copy StarDict dictionaries (read-only lookup data)
COPY wikdict-en-es/ ./wikdict-en-es/
COPY wikdict-es-en/ ./wikdict-es-en/

# Copy the webapp
COPY webapp/ ./webapp/

# Ensure runtime directories exist
RUN mkdir -p webapp/jobs webapp/uploads

# Default Ollama host — points to host machine from inside Docker.
# On Mac (Docker Desktop / OrbStack) host.docker.internal resolves correctly.
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=llama3.1 \
    WEBAPP_PORT=5000 \
    PYTHONPATH=/app

EXPOSE 5000

# Single worker so concurrent jobs don't fight over Ollama.
# Multiple threads inside the worker handle Flask + background job.
CMD ["gunicorn", "--chdir", "webapp", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "0", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
