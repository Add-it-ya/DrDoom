# Two stages, because the things needed to build this image are much larger than the
# things needed to run it. The builder installs dependencies and bakes in the two assets
# that would otherwise be fetched on first request -- the document corpus and the
# embedding model -- so a cold container answers immediately instead of downloading half
# a gigabyte while someone waits.
#
# Torch comes from the CPU-only index. The default wheel carries CUDA libraries worth
# several gigabytes, and nothing here trains on a GPU.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Dependencies first, as their own layer: application code changes far more often than
# the lockfile, and this way editing a module does not reinstall torch.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Bake the corpus. The archive it comes from is large; what survives is a few megabytes
# of extracted documentation, so this happens here rather than in the runtime image.
ENV PATH="/build/.venv/bin:$PATH" \
    PYTHONPATH=/build/src
RUN python -c "from drdoom.rag import corpus; corpus.download()"

# Bake the sentence-transformer weights into the image for the same reason.
ENV HF_HOME=/build/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"


FROM python:3.12-slim-bookworm AS runtime

# Runs as a non-root user. The process reads documentation and writes a sqlite file; it
# has no reason to be able to modify its own code.
RUN useradd --create-home --uid 10001 drdoom

WORKDIR /app

COPY --from=builder --chown=drdoom:drdoom /build/.venv /app/.venv
COPY --from=builder --chown=drdoom:drdoom /build/data/raw/corpus /app/data/raw/corpus
COPY --from=builder --chown=drdoom:drdoom /build/.cache/huggingface /home/drdoom/.cache/huggingface
COPY --chown=drdoom:drdoom src /app/src
COPY --chown=drdoom:drdoom web /app/web
COPY --chown=drdoom:drdoom evals /app/evals
COPY --chown=drdoom:drdoom README.md pyproject.toml /app/

# State is a mount point. A container that loses its suspended investigations on restart
# would undo the durability the graph exists to provide.
RUN mkdir -p /app/state && chown drdoom:drdoom /app/state
VOLUME ["/app/state"]

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/drdoom/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    DRDOOM_ENVIRONMENT=production \
    DRDOOM_LOG_LEVEL=INFO

USER drdoom
EXPOSE 8000

# Checks the application, not just the port: a process that is listening but cannot
# serve is worse than one that is down, because nothing replaces it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "drdoom.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
