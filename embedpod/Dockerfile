# --- Stage 1: Model Collector (DNA Retrieval) ---
FROM python:3.11-slim AS builder
WORKDIR /build

RUN pip install --no-cache-dir huggingface_hub==0.20.*

# We MUST set local_dir_use_symlinks=False so the actual .onnx file 
# is copied in the next stage, rather than a broken symlink.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download( \
        repo_id='nomic-ai/nomic-bert-2048', \
        local_dir='model_artifacts', \
        local_dir_use_symlinks=False, \
        allow_patterns=['configuration_hf_nomic_bert.py','modeling_hf_nomic_bert.py'] \
    )"

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download( \
        repo_id='nomic-ai/nomic-embed-text-v1.5', \
        local_dir='model_artifacts', \
        local_dir_use_symlinks=False, \
        allow_patterns=['onnx/model.onnx','config.json','tokenizer*','special_tokens_map.json'] \
    )"

# --- Stage 2: Production Runtime (The Power Suit) ---
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_VERBOSITY=error

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Installing these ensures the D.R.A.W.N. node has all math libs
RUN pip install --no-cache-dir \
    runpod \
    onnxruntime-gpu \
    transformers \
    numpy

# Copy the actual files (no longer symlinks)
COPY --from=builder /build/model_artifacts /app/model_artifacts
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
