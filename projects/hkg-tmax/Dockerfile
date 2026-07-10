FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/src \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN useradd --create-home --uid 10001 hkg

WORKDIR /workspace
COPY pyproject.toml README.md requirements.txt requirements-dev.txt ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[research]"

COPY config ./config
COPY db ./db
COPY scripts ./scripts

USER hkg
CMD ["python", "-m", "hkg_tmax", "doctor"]
