FROM python:3.14-slim AS builder

WORKDIR /vlrggapi

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && uv pip install --system --no-cache -r requirements.txt \
    && apt-get purge -y --auto-remove gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

FROM python:3.14-slim

WORKDIR /vlrggapi

COPY --from=builder /usr/local /usr/local
COPY api ./api
COPY models ./models
COPY routers ./routers
COPY utils ./utils
COPY main.py .

CMD ["python", "main.py"]
HEALTHCHECK --interval=5s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3001/v2/health', timeout=2)"
