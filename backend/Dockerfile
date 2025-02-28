FROM python:3.11-slim AS builder

WORKDIR /build

# Install necessary system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv and add it to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install uv

# Create virtual environment
RUN uv venv /venv

ENV PATH="/venv/bin:$PATH"

# Copy requirements file
COPY requirements.txt .

# Install dependencies with uv
RUN grep -A 7 "# Core dependencies" requirements.txt > core-deps.txt && \
    grep -A 5 "# Database" requirements.txt > db-deps.txt && \
    grep -A 4 "# Background tasks" requirements.txt > bg-deps.txt && \
    grep -A 9 "# Web parsing" requirements.txt > web-deps.txt && \
    grep -A 12 "# NLP" requirements.txt > nlp-deps.txt && \
    grep -A 3 "# Data processing" requirements.txt > data-deps.txt && \
    grep -A 4 "# HTML/CSS" requirements.txt > html-deps.txt && \
    grep -A 6 "# Utilities" requirements.txt > util-deps.txt

# Install dependencies in groups using uv with retry logic
RUN uv pip install -r core-deps.txt || uv pip install -r core-deps.txt && \
    uv pip install -r db-deps.txt || uv pip install -r db-deps.txt && \
    uv pip install -r bg-deps.txt || uv pip install -r bg-deps.txt && \
    uv pip install -r web-deps.txt || uv pip install -r web-deps.txt && \
    uv pip install -r nlp-deps.txt || uv pip install -r nlp-deps.txt && \
    uv pip install -r data-deps.txt || uv pip install -r data-deps.txt && \
    uv pip install -r html-deps.txt || uv pip install -r html-deps.txt && \
    uv pip install -r util-deps.txt || uv pip install -r util-deps.txt

# Install NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

RUN uv pip install passlib
# Install Playwright separately
RUN uv pip install playwright==1.50.0 && \
    playwright install --with-deps chromium


# Second stage: production image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"

# Install only necessary packages for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    # Playwright dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder image
COPY --from=builder /venv /venv

# Copy Playwright browsers
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Copy application
COPY . .

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Create and set up entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Run migrations if needed\n\
if [ "$1" = "migrate" ]; then\n\
    alembic upgrade head\n\
    exit 0\n\
fi\n\
\n\
# Run celery worker if needed\n\
if [ "$1" = "celery" ]; then\n\
    celery -A app.tasks.worker worker --loglevel=info\n\
    exit 0\n\
fi\n\
\n\
# Run flower if needed\n\
if [ "$1" = "flower" ]; then\n\
    celery -A app.tasks.worker flower --port=5555\n\
    exit 0\n\
fi\n\
\n\
# By default, execute the command passed to docker run\n\
exec "$@"' > /entrypoint.sh

RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]