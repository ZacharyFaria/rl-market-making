FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY configs/ ./configs/
COPY src/ ./src/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["rlmm"]
CMD ["--help"]

