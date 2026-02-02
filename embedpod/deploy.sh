#!/usr/bin/env bash
set -e

ROOT_DIR="$(pwd)"
VERSION_FILE="$ROOT_DIR/VERSION"
IMAGE_NAME="embedpod"

# Read version
if [[ ! -f "$VERSION_FILE" ]]; then
    echo "1.1.0" > "$VERSION_FILE"
fi
VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')
FULL_IMAGE_NAME="${IMAGE_NAME}:v${VERSION}"

echo "Cleaning old build artifacts..."
rm -rf "$ROOT_DIR/build"
mkdir -p "$ROOT_DIR/build/models"

echo "Copying necessary files..."
cp "$ROOT_DIR/Dockerfile" "$ROOT_DIR/build/"
cp "$ROOT_DIR/handler.py" "$ROOT_DIR/build/"
cp "$ROOT_DIR/models/model_q4.onnx" "$ROOT_DIR/build/models/"
cp "$ROOT_DIR/models/tokenizer.json" "$ROOT_DIR/build/models/"

echo "Building Docker image: $FULL_IMAGE_NAME"
docker build -t "$FULL_IMAGE_NAME" "$ROOT_DIR/build"

if [[ -f "$ROOT_DIR/test_input.json" ]]; then
    echo "Running post-build validation (Cold Start Benchmark)..."
    cp "$ROOT_DIR/test_input.json" "$ROOT_DIR/build/"

    time docker run --rm \
        -v "$ROOT_DIR/build/test_input.json":/app/test_input.json:ro \
        "$FULL_IMAGE_NAME" \
        python3 -u handler.py /app/test_input.json

    echo "Validation complete."
else
    echo "No test_input.json found, skipping benchmark."
fi

rm -rf "$ROOT_DIR/build"
echo "Docker image $FULL_IMAGE_NAME built successfully!"
