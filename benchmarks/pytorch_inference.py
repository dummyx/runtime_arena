"""PyTorch Stable Diffusion inference benchmark."""
from __future__ import annotations

import gc, logging, time
from typing import Optional

import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import PNDMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pytorch_inference(
    *,
    model_id: str,
    prompt: str,
    batch_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    output_image_prefix: str,
) -> Optional[float]:
    """Runs diffusion inference with PyTorch diffusers models."""
    logger.info("Starting PyTorch inference benchmark...")

    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(DEVICE)

    vae_scaling_factor = vae.config.scaling_factor
    in_channels = unet.config.in_channels

    prompts = [prompt] * batch_size
    start_time = time.time()

    logger.info("Encoding prompts...")
    text_input_ids = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids).last_hidden_state

    uncond_ids = tokenizer([""] * batch_size, padding="max_length", max_length=text_input_ids.shape[-1], return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_ids).last_hidden_state

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    logger.info("Preparing latents...")
    latents = torch.randn(batch_size, in_channels, height // 8, width // 8, dtype=text_embeddings.dtype, device=DEVICE)
    if isinstance(scheduler.init_noise_sigma, torch.Tensor):
        latents *= scheduler.init_noise_sigma.to(latents.device)
    else:
        latents *= scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps)
    logger.info("Running denoising loop...")

    for t in tqdm(scheduler.timesteps):
        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        n_uncond, n_text = noise_pred.chunk(2)
        noise_pred = n_uncond + guidance_scale * (n_text - n_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    logger.info("Decoding latents...")
    latents = latents / vae_scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample

    inference_time = time.time() - start_time
    logger.info("PyTorch core inference time: %.4fs", inference_time)

    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image.cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
    for i, img in enumerate(image):
        Image.fromarray(img).save(f"{output_image_prefix}_{i}.png")

    del text_encoder, unet, vae, scheduler, tokenizer, latents
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return inference_time
