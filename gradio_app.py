import gradio as gr
from stable_diffusion_pytorch import model_loader, pipeline
import torch
from PIL import Image

dtype = torch.float16
models = model_loader.preload_models('cpu', dtype=dtype)

def generate_image(prompt, uncond_prompt, n_inference_steps, progress=gr.Progress()):
    prompts = [prompt]
    uncond_prompts = [uncond_prompt] if uncond_prompt else None

    progress(0, desc="Generating image...")

    output = pipeline.generate(
        prompts=prompts, 
        uncond_prompts=uncond_prompts,
        height=768, 
        width=768,
        n_inference_steps=n_inference_steps,
        models=models, 
        device='cuda', 
        idle_device='cpu', 
        progress=progress
    )[0]

    return output

gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Enter your prompt", placeholder="Type your prompt here"),
        gr.Textbox(label="Enter negative prompt", placeholder="Type your negative prompt here"),
        gr.Slider(minimum=10, maximum=50, step=1, value=30, label="Number of inference steps")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Stable Diffusion Generator"
).launch(share=True)
