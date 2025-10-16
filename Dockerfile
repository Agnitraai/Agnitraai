# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV AGNITRA_ENV=production

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md setup.py ./ 
COPY agnitra ./agnitra
COPY cli ./cli
COPY docs ./docs

RUN pip install --upgrade pip \
    && pip install .[marketplace]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "agnitra.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

