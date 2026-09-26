# syntax=docker/dockerfile:1.7

# Three stages, two images. The builder has pip's download cache and the
# CPU torch wheel; the runtime has a virtualenv and nothing else. The ui
# stage is separate because Streamlit talks to the API over HTTP and does
# not need torch: shipping 770MB of tensor library to draw a table is the
# kind of thing that happens when one image is "simpler".
#
#   docker build -t joblens .                         the API (last stage, so default)
#   docker build -t joblens-ui --target ui .          the Streamlit UI
#   docker run --rm -p 8000:8000 -e DATABASE_URL=... joblens

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch first, from the CPU index. The default PyPI wheel on Linux drags in
# the CUDA libraries, about 2.5GB, for a container that will never see a
# GPU. Pinning the index here is the single biggest lever on image size.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies before source, so a code change does not reinstall the world.
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Bake the two models the API loads. Without this the first request after
# every cold start waits on a 90MB download from the Hub, and a free tier
# host with an ephemeral disk pays that on every boot.
#
# mkdir first: with BAKE_MODELS=0 nothing else creates /opt/hf, and the
# runtime stage's `COPY --from=builder /opt/hf` fails on a missing path.
ARG BAKE_MODELS=1
RUN mkdir -p /opt/hf && if [ "$BAKE_MODELS" = "1" ]; then \
        python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; \
    fi

# Trim what the runtime never opens: C++ headers, torch's own test binaries
# and every .pyc (PYTHONDONTWRITEBYTECODE means they are not recreated).
RUN SP=/opt/venv/lib/python3.12/site-packages \
    && rm -rf "$SP/torch/include" "$SP/torch/test" \
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} +

FROM python:3.12-slim AS ui

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JOBLENS_API=http://api:8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system joblens \
    && useradd --system --gid joblens --home /app --no-create-home joblens \
    && pip install "streamlit>=1.37" "httpx>=0.27" "pandas>=2.2" \
    && find /usr/local/lib/python3.12 -type d -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /app
COPY --chown=joblens:joblens app.py ./
USER joblens
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]


# The API. Last on purpose: a plain `docker build .` must produce this one.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    # Production defaults. Every one of these is overridable from the
    # environment, and none of them is a secret.
    APP_ENV=production \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO

# With the models baked in, startup must never touch huggingface.co: the
# library otherwise HEADs the Hub for every file even on a warm cache, and
# a Hub outage or an egress rule turns into a slow or stalled boot.
ARG BAKE_MODELS=1
ENV HF_HUB_OFFLINE=${BAKE_MODELS}

# curl for the healthcheck, nothing else. No compiler, no git.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system joblens \
    && useradd --system --gid joblens --home /app --no-create-home joblens

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf
COPY --chown=joblens:joblens migrations ./migrations
COPY --chown=joblens:joblens prompts ./prompts
COPY --chown=joblens:joblens app.py ./
COPY --chown=joblens:joblens src ./src

# joblens is installed into the venv, but db.migrate() and prompts.load()
# resolve their files relative to the source tree, so the tree is here too.
ENV PYTHONPATH=/app/src

USER joblens
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "joblens", "serve", "--host", "0.0.0.0", "--port", "8000"]
