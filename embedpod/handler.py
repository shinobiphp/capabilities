#!/usr/bin/env python3
import os
import sys
import json
import time
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "/app/model"
MODEL_FILE = os.path.join(MODEL_DIR, "model_q4.onnx")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")

# Use GPU if available and ONNXRuntime built with CUDA, else CPU
EP = ["CUDAExecutionProvider"] if "gpu" in sys.argv[0].lower() else ["CPUExecutionProvider"]

print(f"EmbedPod - By Shinobi: Init ({'GPU' if 'CUDAExecutionProvider' in EP else 'CPU'})...")

# -----------------------------
# Load tokenizer
# -----------------------------
start = time.time()
TOKENIZER = Tokenizer.from_file(TOKENIZER_FILE)
print("Tokenizer loaded.")

# -----------------------------
# Load ONNX model session
# -----------------------------
SESSION = ort.InferenceSession(MODEL_FILE, providers=EP)
print("ONNX model session ready.")

print(f"EmbedPod - By Shinobi: Ready. (Load time: {time.time()-start:.2f}s)")

# -----------------------------
# Tokenization + input prep
# -----------------------------
def tokenize(text: str):
    encoding = TOKENIZER.encode(text)
    tokens = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
    }

    # Ensure all required model inputs exist
    model_inputs = [i.name for i in SESSION.get_inputs()]
    if "token_type_ids" in model_inputs and "token_type_ids" not in tokens:
        tokens["token_type_ids"] = np.zeros_like(tokens["input_ids"], dtype=np.int64)

    return tokens

# -----------------------------
# Run embedding
# -----------------------------
def run_embedding(text: str):
    tokens = tokenize(text)
    return SESSION.run(None, tokens)[0][0].tolist()  # return first batch as list

# -----------------------------
# Command-line / file support
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "r") as f:
            payload = json.load(f)
        text = payload.get("text", "")
    elif len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "Hello, world!"

    start = time.time()
    embeddings = run_embedding(text)
    elapsed = time.time() - start
    print(json.dumps({"embeddings": embeddings, "compute_time": elapsed}, ensure_ascii=False))
