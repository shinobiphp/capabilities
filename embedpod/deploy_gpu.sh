#!/usr/bin/env bash
set -e

ROOT="$(pwd)"
VERSION=$(cat "$ROOT/VERSION" 2>/dev/null || echo "1.1.0")
IMAGE="embedpod-gpu:v$VERSION"

rm -rf "$ROOT/build"
mkdir -p "$ROOT/build/models"

cp "$ROOT/Dockerfile.gpu" "$ROOT/build/Dockerfile"
cp "$ROOT/handler.py" "$ROOT/build/"
cp "$ROOT/models/model_q4.onnx" "$ROOT/build/models/"
cp "$ROOT/models/tokenizer.json" "$ROOT/build/models/"

echo "Building GPU Docker image $IMAGE..."
docker build -t "$IMAGE" "$ROOT/build"

echo "GPU image built successfully: $IMAGE"
rm -rf "$ROOT/build"
