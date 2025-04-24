"""IREE Stable Diffusion inference."""
from __future__ import annotations

import os, time, logging, gc
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import PNDMScheduler
from transformers import CLIPTokenizer

import iree.runtime as ireert
from iree.runtime import FunctionInvoker

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _multiply_sigma(latents: np.ndarray, sigma):
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.item()
    return latents * sigma


def run_iree_inference(
    model_id: str,
    vmfb_dir: str,
    prompt: str,
    batch_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    output_image_prefix: str,
    iree_target_backend: str = "cuda",
    iree_device: str = "cuda",
) -> Optional[float]:
    """Runs SD inference using compiled IREE VMFB modules."""
    logger.info("Starting IREE inference...")

    
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    vae_scaling_factor, unet_channels = 0.18215, 4

    
    names = {
        "unet": f"unet_{iree_target_backend}.vmfb",
        "text": f"text_encoder_{iree_target_backend}.vmfb",
        "vae": f"vae_decoder_{iree_target_backend}.vmfb",
    }
    paths = {k: os.path.join(vmfb_dir, v) for k, v in names.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        logger.error("Missing VMFB files – aborting IREE benchmark.")
        return None

    
    cfg = ireert.Config(driver_name=iree_device)
    ctxs = {k: ireert.SystemContext(config=cfg) for k in ("unet", "text", "vae")}
    device = ireert.get_first_device()
    modules = {}
    for key in ctxs:
        with open(paths[key], "rb") as f:
            modules[key] = ireert.VmModule.from_flatbuffer(ctxs[key].instance, f.read())
            ctxs[key].add_vm_module(modules[key])
    invokers = {
        "unet": FunctionInvoker(ctxs["unet"].vm_context, device, modules["unet"].lookup_function("main_graph")),
        "text": FunctionInvoker(ctxs["text"].vm_context, device, modules["text"].lookup_function("main_graph")),
        "vae": FunctionInvoker(ctxs["vae"].vm_context, device, modules["vae"].lookup_function("main_graph")),
    }

    
    prompts = [prompt] * batch_size
    text_in = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    txt_ids_np = text_in.input_ids.numpy()
    text_emb = invokers["text"](txt_ids_np)[0].to_host()

    uncond_ids_np = tokenizer([""] * batch_size, padding="max_length", max_length=text_in.input_ids.shape[-1], return_tensors="pt").input_ids.numpy()
    uncond_emb = invokers["text"](uncond_ids_np)[0].to_host()
    text_embeddings = np.concatenate([uncond_emb, text_emb])

    
    latents = np.random.randn(batch_size, unet_channels, height // 8, width // 8).astype("float32")
    latents = _multiply_sigma(latents, scheduler.init_noise_sigma)

    
    scheduler.set_timesteps(num_inference_steps)
    text_t = torch.from_numpy(text_embeddings)
    latents_t = torch.from_numpy(latents)

    for t in tqdm(scheduler.timesteps):
        l_in = torch.cat([latents_t] * 2)
        l_in = scheduler.scale_model_input(l_in, t).numpy()
        noise_pred1, noise_pred2 = invokers["unet"](l_in, np.array(t, dtype=np.float32), text_embeddings)
        noise_pred = torch.cat([torch.from_numpy(noise_pred1), torch.from_numpy(noise_pred2)])
        noise_pred_u, noise_pred_c = noise_pred.chunk(2)
        noise_pred = noise_pred_u + guidance_scale * (noise_pred_c - noise_pred_u)
        latents_t = scheduler.step(noise_pred, t, latents_t).prev_sample

    latents = latents_t.numpy()

    
    latents = latents / vae_scaling_factor
    image = invokers["vae"](latents)[0].to_host()
    image = (image / 2 + 0.5).clip(0, 1)
    image = (image * 255).round().astype("uint8")
    image = np.transpose(image, (0, 2, 3, 1))

    for i, img in enumerate(image):
        Image.fromarray(img).save(f"{output_image_prefix}_{i}.png")

    gc.collect()
    return None
