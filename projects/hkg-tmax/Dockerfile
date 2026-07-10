FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY code ./code
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[research,dev]"

COPY . .
ENV PYTHONPATH=/workspace/code/src:/workspace

CMD ["bash"]
