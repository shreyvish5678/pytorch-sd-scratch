import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from . import Tokenizer
from . import KLMSSampler, KEulerSampler, KEulerAncestralSampler
from . import util
from . import model_loader
import gradio as gr


def generate(
        prompts,
        uncond_prompts=None,
        input_images=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        height=512,
        width=512,
        sampler="k_lms",
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        progress=None   
):
    with torch.no_grad():
        if not isinstance(prompts, (list, tuple)) or not prompts:
            raise ValueError("prompts must be a non-empty list or tuple")

        if uncond_prompts and not isinstance(uncond_prompts, (list, tuple)):
            raise ValueError("uncond_prompts must be a non-empty list or tuple if provided")
        if uncond_prompts and len(prompts) != len(uncond_prompts):
            raise ValueError("length of uncond_prompts must be same as length of prompts")
        uncond_prompts = uncond_prompts or [""] * len(prompts)

        if input_images and not isinstance(uncond_prompts, (list, tuple)):
            raise ValueError("input_images must be a non-empty list or tuple if provided")
        if input_images and len(prompts) != len(input_images):
            raise ValueError("length of input_images must be same as length of prompts")
        if not 0 < strength < 1:
            raise ValueError("strength must be between 0 and 1")

        if height % 8 or width % 8:
            raise ValueError("height and width must be a multiple of 8")

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        tokenizer = Tokenizer()
        clip = models.get('clip') or model_loader.load_clip(device)
        clip.to(device)

        dtype = clip.embedding.position_value.dtype
        if do_cfg:
            cond_tokens = tokenizer.encode_batch(prompts)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.encode_batch(uncond_prompts)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.encode_batch(prompts)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip)
        del tokenizer, clip

        if sampler == "k_lms":
            sampler = KLMSSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler":
            sampler = KEulerSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler_ancestral":
            sampler = KEulerAncestralSampler(n_inference_steps=n_inference_steps,
                                             generator=generator)
        else:
            raise ValueError(
                "Unknown sampler value %s. "
                "Accepted values are {k_lms, k_euler, k_euler_ancestral}"
                % sampler
            )

        noise_shape = (len(prompts), 4, height // 8, width // 8)

        if input_images:
            encoder = models.get('encoder') or model_loader.load_encoder(device)
            encoder.to(device)
            processed_input_images = []
            for input_image in input_images:
                if type(input_image) is str:
                    input_image = Image.open(input_image)

                input_image = input_image.resize((width, height))
                input_image = np.array(input_image)
                input_image = torch.tensor(input_image, dtype=dtype)
                input_image = util.rescale(input_image, (0, 255), (-1, 1))
                processed_input_images.append(input_image)
            input_images_tensor = torch.stack(processed_input_images).to(device)
            input_images_tensor = util.move_channel(input_images_tensor, to="first")

            _, _, height, width = input_images_tensor.shape
            noise_shape = (len(prompts), 4, height // 8, width // 8)

            encoder_noise = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            latents = encoder(input_images_tensor, encoder_noise)

            latents_noise = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            sampler.set_strength(strength=strength)
            latents += latents_noise * sampler.initial_scale

            to_idle(encoder)
            del encoder, processed_input_images, input_images_tensor, latents_noise
        else:
            latents = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            latents *= sampler.initial_scale

        diffusion = models.get('diffusion') or model_loader.load_diffusion(device)
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = util.get_time_embedding(timestep, dtype).to(device)

            input_latents = latents * sampler.get_input_scale()
            if do_cfg:
                input_latents = input_latents.repeat(2, 1, 1, 1)
            output = diffusion(input_latents, context, time_embedding)
            if do_cfg:
                output_cond, output_uncond = output.chunk(2)
                output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(latents, output)
            progress((i + 1) / len(sampler.timesteps), desc="Generating image...")

        progress(1, desc="Decoding image...")

        to_idle(diffusion)
        del diffusion

        decoder = models.get('decoder') or model_loader.load_decoder(device)
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)
        del decoder

        images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = util.move_channel(images, to="last")
        images = images.to('cpu', torch.uint8).numpy()

        return [Image.fromarray(image) for image in images]