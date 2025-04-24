"""Optimum Stable Diffusion ONNX Runtime inference benchmark."""
from __future__ import annotations

import gc, logging, time
from typing import Optional

from PIL import Image
from optimum.onnxruntime import ORTStableDiffusionPipeline
import onnxruntime as ort

logger = logging.getLogger(__name__)


def _select_provider() -> str:
    avail = ort.get_available_providers()
    for p in ("CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"):
        if p in avail:
            return p
    return "CPUExecutionProvider"


def run_optimum_onnx_inference(
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
    """Runs inference using Optimum/ONNX Runtime pipeline."""
    provider = _select_provider()
    logger.info("Optimum ORT provider: %s", provider)

    t0 = time.time()
    try:
        pipe = ORTStableDiffusionPipeline.from_pretrained(model_id, provider=provider, export=True)
    except Exception as e:
        logger.error("Optimum pipeline load failed: %s", e)
        return None
    logger.info("Pipeline loaded in %.2fs", time.time() - t0)

    t_start = time.time()
    imgs = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
    infer_time = time.time() - t_start

    for i, img in enumerate(imgs):
        img.save(f"{output_image_prefix}_onnx_{i}.png")

    del pipe, imgs
    gc.collect()
    return infer_time
