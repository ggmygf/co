import torch
import gradio as gr
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Load model
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16
).to("cuda")

def process_edit(img1, img2, prompt):
    if img1 is None or img2 is None:
        return None
    
    # Force resize to avoid math errors
    t_img = img1.convert("RGB").resize((1024, 1024))
    r_img = img2.convert("RGB").resize((1024, 1024))

    print("--- STARTING CALCULATION ---") # LOG CHECK
    output = pipe(
        image=[t_img, r_img],
        prompt=prompt,
        negative_prompt=" ", 
        num_inference_steps=40,
        true_cfg_scale=4.0,
        height=1024,
        width=1024
    ).images[0]
    
    print("--- DONE! SENDING IMAGE ---") # IF YOU SEE THIS, THE AI WORKED
    return output

with gr.Blocks() as demo:
    gr.Markdown("### Qwen Fixed Display")
    with gr.Row():
        with gr.Column():
            slot1 = gr.Image(label="TARGET", type="pil")
            slot2 = gr.Image(label="REF", type="pil")
            prompt = gr.Textbox(label="Instruction", value="Change background to a forest")
            btn = gr.Button("RUN")
        with gr.Column():
            # Added interactive=False to make it a pure output slot
            result = gr.Image(label="RESULT", interactive=False)

    btn.click(fn=process_edit, inputs=[slot1, slot2, prompt], outputs=result)

# debug=True forces logs to show up even if it crashes
demo.queue().launch(server_name="0.0.0.0", server_port=8000, debug=True)
