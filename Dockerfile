FROM python:3.11-slim

WORKDIR /vlrggapi

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential curl \
    && uv pip install --system --no-cache -r requirements.txt \
    && apt-get purge -y --auto-remove gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "main.py"]
HEALTHCHECK --interval=5s --timeout=3s CMD curl --fail http://127.0.0.1:3001/health || exit 1
