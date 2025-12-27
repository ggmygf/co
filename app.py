import torch
import gradio as gr
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Load the model into the H100 (80GB VRAM)
# Using bfloat16 for speed and memory efficiency
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16
).to("cuda")

def edit_image(target_img, ref_img, prompt, steps, cfg):
    if target_img is None:
        return None
    
    # Qwen 2509 supports 1 to 3 images. 
    # We pass [Target, Reference] as a list.
    imgs = [target_img]
    if ref_img is not None:
        imgs.append(ref_img)
    
    output = pipe(
        image=imgs,
        prompt=prompt,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
    ).images[0]
    
    return output

# Build the Web UI
with gr.Blocks() as demo:
    gr.Markdown("# Qwen Image Edit 2509 (H100 Edition)")
    
    with gr.Row():
        with gr.Column():
            target = gr.Image(label="1. Upload Image to Fix (Target)", type="pil")
            reference = gr.Image(label="2. Upload Lighting Reference (Optional)", type="pil")
            prompt = gr.Textbox(
                label="Instructions", 
                value="transfer the professional lighting and skin texture from image 2 to the person in image 1"
            )
            btn = gr.Button("Fix Image", variant="primary")
        
        with gr.Column():
            result = gr.Image(label="Result")
            
    with gr.Accordion("Advanced Settings", open=False):
        steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Steps")
        cfg = gr.Slider(minimum=1, maximum=10, value=4.0, step=0.5, label="CFG Scale")

    btn.click(fn=edit_image, inputs=[target, reference, prompt, steps, cfg], outputs=result)

# Run on port 8000 for Koyeb
demo.launch(server_name="0.0.0.0", server_port=8000)
