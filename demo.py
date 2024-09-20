from stable_diffusion_pytorch import model_loader, pipeline
import torch
dtype = torch.float16
models = model_loader.preload_models('cpu', dtype=dtype)

prompt = "A painting of a cat"  
uncond_prompt = ""  
size = 512
n_inference_steps = 30

prompts = [prompt]
uncond_prompts = [uncond_prompt] if uncond_prompt else None

output = pipeline.generate(prompts=prompts, height=size, width=size,
                  n_inference_steps=n_inference_steps,
                  models=models, device='cuda', idle_device='cpu')[0]

output.save("output.png")