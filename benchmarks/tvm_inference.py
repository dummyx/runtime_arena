"""TVM Stable Diffusion inference benchmark wrapper.

Relies on the `TVMStableDiffusionPipeline` implementation provided in
`tvm_diffusers_e2e.py`. This module simply instantiates the pipeline and
measures the time to generate images for a prompt, matching the signature
of the other benchmark helpers so it can be called from `benchmark.py`.
"""
from __future__ import annotations

import gc
import logging
import time
from typing import List, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)

# Lazy import to avoid heavy TVM startup if benchmark is skipped
try:
    from tvm_diffusers_e2e import TVMStableDiffusionPipeline
except Exception as _imp_err:  # pragma: no cover
    TVMStableDiffusionPipeline = None  # type: ignore
    logger.warning("Could not import TVMStableDiffusionPipeline: %s", _imp_err)


def run_tvm_inference(
    *,
    model_id: str,
    prompt: str,
    batch_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    output_image_prefix: str,
    tvm_device: str = "cpu",
    tuning_trials: int = 0,
    target_str: str = "llvm -num-cores 16 -mcpu=generic",
) -> Optional[float]:
    """Runs Stable Diffusion inference using TVM‑compiled models.

    Returns the time taken for the *pipeline call* (does NOT include initial
    compilation time, which can be very large). If compilation or inference
    fails, returns ``None``.
    """

    if TVMStableDiffusionPipeline is None:
        logger.error("TVMStableDiffusionPipeline import failed – skipping TVM benchmark.")
        return None

    # ------------------------------------------------------------------
    logger.info("Instantiating TVM pipeline (device=%s, trials=%d)…", tvm_device, tuning_trials)
    t0 = time.time()
    try:
        pipeline = TVMStableDiffusionPipeline(
            model_id=model_id,
            device=tvm_device,
            target_str=target_str,
            tuning_trials=tuning_trials,
        )
    except Exception as e:
        logger.error("TVM pipeline initialization failed: %s", e)
        return None
    compile_time = time.time() - t0
    logger.info("TVM pipeline ready (initial compile %.2fs).", compile_time)

    # ------------------------------------------------------------------
    prompts: Union[str, List[str]] = prompt if batch_size == 1 else [prompt] * batch_size
    logger.info("Running TVM inference (steps=%d)…", num_inference_steps)
    t_infer_start = time.time()
    try:
        images = pipeline(
            prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    except Exception as e:
        logger.error("TVM pipeline call failed: %s", e)
        return None
    infer_time = time.time() - t_infer_start
    logger.info("TVM core inference time: %.4fs", infer_time)

    # ------------------------------------------------------------------
    if images:
        logger.info("Saving generated TVM image(s)…")
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                try:
                    # If ndarray
                    from PIL import Image as _PILImage

                    img = _PILImage.fromarray(img)
                except Exception:
                    logger.warning("Could not convert image %d to PIL — skipping save.", i)
                    continue
            img.save(f"{output_image_prefix}_tvm_{i}.png")

    # Clean‑up ----------------------------------------------------------
    del pipeline, images  # type: ignore
    gc.collect()

    return infer_time
