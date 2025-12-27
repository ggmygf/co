import torch
import gradio as gr
import requests
from PIL import Image
from io import BytesIO
from diffusers import QwenImageEditPlusPipeline

# 1. LOAD MODEL (Full bfloat16 for H100)
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16
).to("cuda")

def process_edit(target_url, ref_url, prompt, neg_prompt):
    try:
        # Download images from the links you provide
        t_resp = requests.get(target_url, timeout=10)
        r_resp = requests.get(ref_url, timeout=10)
        
        img1 = Image.open(BytesIO(t_resp.content)).convert("RGB")
        img2 = Image.open(BytesIO(r_resp.content)).convert("RGB")
        
        # 2. MATCH SIZES (Prevents the crash you experienced)
        # We target 1024x1024 as the model's sweet spot
        img1 = img1.resize((1024, 1024), Image.LANCZOS)
        img2 = img2.resize((1024, 1024), Image.LANCZOS)

        # 3. RUN THE EDIT
        # If Slot 1 and Slot 2 are the same pic, it acts as a "Self-Reference" 
        # which makes the edit much cleaner and holds the identity better.
        output = pipe(
            image=[img1, img2],
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=40,
            true_cfg_scale=4.0,  # Fixed: Now has negative_prompt to reference
            height=1024,
            width=1024
        ).images[0]
        
        return output
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None

# 4. THE UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Qwen Multi-Image Editor (H100 Optimized)")
    
    with gr.Row():
        with gr.Column():
            url1 = gr.Textbox(label="SLOT 1: TARGET IMAGE URL (The Base)", placeholder="Paste .jpg/.png link...")
            url2 = gr.Textbox(label="SLOT 2: REFERENCE IMAGE URL (The Style/Identity)", placeholder="Paste .jpg/.png link...")
            
            prompt_input = gr.Textbox(
                label="Instruction", 
                value="Using the identity/details from image 2, change the target image 1 by: [YOUR EDIT HERE]"
            )
            
            neg_input = gr.Textbox(
                label="Negative Prompt (Prevents Crashes)", 
                value="low quality, blurry, distorted, deformed, out of focus"
            )
            
            run_btn = gr.Button("GENERATE EDIT", variant="primary")
        
        with gr.Column():
            result_display = gr.Image(label="RESULT VIEW")

    run_btn.click(
        fn=process_edit, 
        inputs=[url1, url2, prompt_input, neg_input], 
        outputs=result_display
    )

# Koyeb needs 0.0.0.0 to expose the port
demo.launch(server_name="0.0.0.0", server_port=8000)
