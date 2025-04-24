"""Direct ONNX Runtime Stable Diffusion inference benchmark."""
from __future__ import annotations

import gc, logging, os, time
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm.auto import tqdm

from diffusers import PNDMScheduler
from transformers import CLIPTokenizer

logger = logging.getLogger(__name__)


def _provider() -> str:
    avail = ort.get_available_providers()
    for p in ("CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"):
        if p in avail:
            return p
    return "CPUExecutionProvider"


def _mul_sigma(latents: np.ndarray, sigma):
    if isinstance(sigma, np.ndarray):
        return latents * sigma
    return latents * float(sigma)


def run_direct_onnx_inference(
    *,
    model_dir: str,
    model_id: str,
    prompt: str,
    batch_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    output_image_prefix: str,
) -> Optional[float]:
    """Runs inference calling ONNX Runtime sessions directly."""
    provider = _provider()
    logger.info("ONNX Runtime provider: %s", provider)

    required = {k: os.path.join(model_dir, f"{k}.onnx") for k in ("text_encoder", "unet", "vae_decoder")}
    if not all(os.path.exists(p) for p in required.values()):
        logger.error("Missing ONNX models in %s", model_dir)
        return None

    sess_opt = ort.SessionOptions()
    text_sess = ort.InferenceSession(required["text_encoder"], sess_options=sess_opt, providers=[provider])
    unet_sess = ort.InferenceSession(required["unet"], sess_options=sess_opt, providers=[provider])
    vae_sess = ort.InferenceSession(required["vae_decoder"], sess_options=sess_opt, providers=[provider])

    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    text_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np").input_ids
    text_emb = text_sess.run(None, {"input_ids": text_ids.astype(np.int64)})[0]

    uncond_ids = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, return_tensors="np").input_ids
    uncond_emb = text_sess.run(None, {"input_ids": uncond_ids.astype(np.int64)})[0]
    text_embeddings = np.concatenate([uncond_emb] * batch_size + [text_emb] * batch_size)

    latents = np.random.randn(batch_size, 4, height // 8, width // 8).astype("float32")
    latents = _mul_sigma(latents, scheduler.init_noise_sigma)

    scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(scheduler.timesteps):
        inp = np.concatenate([latents] * 2)
        inp = scheduler.scale_model_input(inp, t if isinstance(t, int) else int(t))
        noise_pred = unet_sess.run(None, {"sample": inp.astype("float32"), "timestep": np.array([t]).astype("float32"), "encoder_hidden_states": text_embeddings.astype("float32")})[0]
        u, c = np.split(noise_pred, 2)
        noise_pred = u + guidance_scale * (c - u)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / 0.18215
    img = vae_sess.run(None, {"latent_sample": latents.astype("float32")})[0]
    img = (img / 2 + 0.5).clip(0, 1)
    img = (img * 255).round().astype("uint8")
    img = np.transpose(img, (0, 2, 3, 1))

    for i, im in enumerate(img):
        Image.fromarray(im).save(f"{output_image_prefix}_{i}.png")

    gc.collect()
    return None
