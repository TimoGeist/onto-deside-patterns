FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.6.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject and poetry.lock first
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-interaction --no-ansi --no-root --no-dev

# Copy the frontend code
COPY . .
COPY .streamlit ./.streamlit

# Final image
FROM python:3.11-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app /app

# Install any system packages needed by frontend (e.g., OCR libraries)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Run streamlit on port 8501
CMD ["streamlit", "run", "app_streamlit.py", "--server.address=0.0.0.0"]
