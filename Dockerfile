# syntax=docker/dockerfile:1

#  Stage 1: shared base with Python dependencies 
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

#  Stage 2: trainer 
# Used by docker compose as the one-shot "trainer" service. It writes the
# model into /app/artifacts, which compose mounts as a shared volume.
FROM base AS trainer
ENV MODEL_DIR=/app/artifacts DATA_DIR=/app/data
COPY data/ data/
COPY src/train.py src/
CMD ["python", "src/train.py"]

#  Stage 3: train at build time
# Lets `docker build` + `docker run` work on their own, without compose.
FROM trainer AS model_build
RUN python src/train.py

#  Stage 4: serving 
FROM base AS serving
ENV MODEL_DIR=/app/artifacts PORT=5000
RUN useradd --create-home --uid 1000 sommelier
COPY src/app.py src/db.py ./
COPY src/templates/ templates/
COPY src/static/ static/
COPY --from=model_build /app/artifacts/ artifacts/
RUN chown -R sommelier:sommelier /app
USER sommelier

EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health', timeout=2)"

#  preload loads the model and creates the DB table once, before workers fork
CMD ["sh", "-c", "gunicorn --preload --bind 0.0.0.0:${PORT} --workers 2 --access-logfile - app:app"]
