import torch
import gradio as gr
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Load the model (Optimized for H100)
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16
).to("cuda")

def process_edit(target_img, ref_img, prompt):
    if target_img is None or ref_img is None:
        return None
    
    # Qwen-2509 takes a list of images
    # Image 1 = Target (to be changed), Image 2 = Reference (the style/light source)
    inputs = {
        "image": [target_img, ref_img],
        "prompt": prompt,
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
    }
    
    with torch.inference_mode():
        output = pipe(**inputs)
    return output.images[0]

# Build the UI
with gr.Blocks(title="Qwen Image Edit 2509") as demo:
    gr.Markdown("# Qwen Image Edit - Portrait & Style Fixer")
    
    with gr.Row():
        with gr.Column():
            # CLEAR SLOT NAMES AS REQUESTED
            input_target = gr.Image(label="SLOT 1: TARGET IMAGE (The one to fix)", type="pil")
            input_ref = gr.Image(label="SLOT 2: REFERENCE IMAGE (The good lighting/style)", type="pil")
            prompt_text = gr.Textbox(
                label="Instruction", 
                placeholder="Example: Apply the lighting and skin texture from Slot 2 to the person in Slot 1",
                value="Transfer the professional lighting from the second image to the person in the first image."
            )
            submit_btn = gr.Button("Fix Image", variant="primary")
        
        with gr.Column():
            output_view = gr.Image(label="RESULT (View & Download)")

    submit_btn.click(
        fn=process_edit, 
        inputs=[input_target, input_ref, prompt_text], 
        outputs=output_view
    )

demo.launch(server_name="0.0.0.0", server_port=8000)
