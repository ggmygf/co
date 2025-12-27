FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python and Git
RUN apt-get update && apt-get install -y python3 python3-pip git

# Install the "Legit" dependencies
# 1. Latest Diffusers from source (Required for Qwen-Edit-Plus)
# 2. Transformers and Accelerate
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install git+https://github.com/huggingface/diffusers.git
RUN pip3 install transformers accelerate safetensors pillow gradio

WORKDIR /app
COPY app.py .

# Expose port 8000 for the Gradio UI
EXPOSE 8000

CMD ["python3", "app.py"]
