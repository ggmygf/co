# Use the official vLLM image as base
FROM vllm/vllm-openai:latest

# Set environment variables for the model
ENV MODEL_NAME="RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8"

# Overwrite the entrypoint to ensure it uses your specific flags
ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model ${MODEL_NAME} \
    --served-model-name mixtral \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
