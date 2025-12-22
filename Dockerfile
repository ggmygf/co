FROM vllm/vllm-openai:latest
ENV MODEL_NAME="RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8"
ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --served-model-name mixtral \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92
