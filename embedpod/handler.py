import runpod
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import os

MODEL_PATH = "/app/model_artifacts/onnx/model.onnx"
TOKENIZER_PATH = "/app/model_artifacts"

print("[D.R.A.W.N.] Initializing embedding runtime")

# Tokenizer (required for Nomic custom layers)
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    trust_remote_code=True
)

# ONNX Runtime session tuning for small GPU pods
session_options = ort.SessionOptions()
session_options.enable_mem_pattern = False
session_options.enable_cpu_mem_arena = False
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=session_options,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

def handler(job):
    try:
        job_input = job.get("input", {})
        text = job_input.get("text")

        if not text:
            return {"error": "N.O.N.O. Missing 'text' input"}

        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="np"
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        outputs = session.run(None, onnx_inputs)
        last_hidden_state = outputs[0]

        attention_mask = onnx_inputs["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)

        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        embeddings = (sum_embeddings / sum_mask).tolist()

        return {"embeddings": embeddings}

    except Exception as e:
        return {"error": f"D.A.R.K.E.S.T. Failure: {str(e)}"}

runpod.serverless.start({"handler": handler})
