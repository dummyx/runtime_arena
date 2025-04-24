import os
import argparse
import logging

import torch

from benchmarks import (
    run_iree_inference,
    run_pytorch_inference,
    run_optimum_onnx_inference,
    run_direct_onnx_inference,
    run_tvm_inference,
)

# ---------------------------------------------------------------------------
# Configuration defaults (overridable via CLI) -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

VMFB_DIR = "./models" 
PROMPT = "a photo of an astronaut riding a horse on the moon"
BATCH_SIZE = 1
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
OUTPUT_IMAGE_PREFIX = "iree_generated_image" 

RUN_BENCHMARK = True
PYTORCH_OUTPUT_IMAGE_PREFIX = "pytorch_generated_image"

IREE_TARGET_BACKEND = "llvm-cpu"
IREE_DEVICE = "local-task"

if __name__ == "__main__":
    print("--- Stable Diffusion IREE Pipeline ---")

    parser = argparse.ArgumentParser(description="Benchmark Stable Diffusion Inference (PyTorch, ONNX Runtime, IREE, TVM).")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Hugging Face model ID for tokenizer/scheduler.")
    parser.add_argument("--model_dir", type=str, default=VMFB_DIR, help="Directory containing ONNX/VMFB models.")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Inference prompt.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for inference.")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Image height.")
    parser.add_argument("--width", type=int, default=WIDTH, help="Image width.")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE, help="Guidance scale.")
    parser.add_argument("--iree_target", type=str, default=IREE_TARGET_BACKEND, help="IREE target backend identifier (used for VMFB filename and runtime driver).")
    parser.add_argument("--iree_device", type=str, default=IREE_DEVICE, help="IREE device identifier (e.g., local-task, local-cpu, local-gpu).")
    parser.add_argument("--skip_pytorch", action='store_true', help="Skip PyTorch benchmark.")
    parser.add_argument("--skip_onnx", action='store_true', help="Skip Optimum ONNX benchmark.")
    parser.add_argument("--skip_iree", action='store_true', help="Skip IREE benchmark.")
    parser.add_argument("--skip_direct_onnx", action='store_true', help="Skip Direct ONNX Runtime benchmark.")
    parser.add_argument("--skip_tvm", action='store_true', help="Skip TVM benchmark.")

    args = parser.parse_args()

    # This file now only orchestrates benchmarks; heavy logic lives in benchmarks package.
    model_dir = args.model_dir
    iree_target = args.iree_target
    iree_device = args.iree_device

    # Check for IREE VMFB files
    if not args.skip_iree:
        unet_vmfb_path = os.path.join(model_dir, f'unet_{iree_target}.vmfb')
        text_encoder_vmfb_path = os.path.join(model_dir, f'text_encoder_{iree_target}.vmfb')
        vae_decoder_vmfb_path = os.path.join(model_dir, f'vae_decoder_{iree_target}.vmfb')

        print(f"Checking for IREE VMFB files in '{model_dir}' for target '{iree_target}':")
        print(f" - {os.path.basename(unet_vmfb_path)}")
        print(f" - {os.path.basename(text_encoder_vmfb_path)}")
        print(f" - {os.path.basename(vae_decoder_vmfb_path)}")

        if not all(os.path.exists(p) for p in [unet_vmfb_path, text_encoder_vmfb_path, vae_decoder_vmfb_path]):
            logger.error(f"ERROR: Compiled VMFB models not found in {model_dir} for target '{iree_target}'.")
            logger.error("Please ensure you have exported/compiled the models using 'export_onnx.py' with the correct --output_dir and --iree_target.")
            exit(1)
        else:
            logger.info(f"Found required IREE VMFB models.")

    # Run benchmarks
    inference_time_iree = None
    if not args.skip_iree:
        inference_time_iree = run_iree_inference(
            model_id=args.model_id, 
            vmfb_dir=model_dir, 
            prompt=args.prompt, 
            batch_size=args.batch_size, 
            height=args.height, 
            width=args.width, 
            num_inference_steps=args.steps, 
            guidance_scale=args.guidance, 
            output_image_prefix="iree_generated_image", 
            iree_target_backend=iree_target, 
            iree_device=iree_device
        )

    inference_time_pytorch = None
    if not args.skip_pytorch:
        print("\n--- Running PyTorch Inference for Benchmark ---")
        inference_time_pytorch = run_pytorch_inference(
            model_id=args.model_id, 
            prompt=args.prompt, 
            batch_size=args.batch_size, 
            height=args.height, 
            width=args.width, 
            num_inference_steps=args.steps, 
            guidance_scale=args.guidance, 
            output_image_prefix=PYTORCH_OUTPUT_IMAGE_PREFIX
        )

    inference_time_optimum_onnx = None
    if not args.skip_onnx:
        print("\n--- Running Optimum ONNX Runtime Inference for Benchmark ---")
        inference_time_optimum_onnx = run_optimum_onnx_inference(
            model_id=args.model_id, 
            prompt=args.prompt, 
            batch_size=args.batch_size, 
            height=args.height, 
            width=args.width, 
            num_inference_steps=args.steps, 
            guidance_scale=args.guidance, 
            output_image_prefix="optimum_generated_image" 
        )

    inference_time_direct_onnx = None
    if not args.skip_direct_onnx:
        print("\n--- Running Direct ONNX Runtime Inference for Benchmark ---")
        if os.path.exists(os.path.join(model_dir, "text_encoder.onnx")) and \
           os.path.exists(os.path.join(model_dir, "unet.onnx")) and \
           os.path.exists(os.path.join(model_dir, "vae_decoder.onnx")):

            inference_time_direct_onnx = run_direct_onnx_inference(
                model_dir=model_dir,
                model_id=args.model_id, 
                prompt=args.prompt,
                batch_size=args.batch_size,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                output_image_prefix="direct_onnx_generated_image" 
            )
        else:
            logger.warning(f"Direct ONNX models not found in expected subdirectories within {model_dir}. Skipping direct ONNX benchmark.")
            logger.warning(f"Expected: {model_dir}/text_encoder/model.onnx, {model_dir}/unet/model.onnx, {model_dir}/vae_decoder/model.onnx")

    inference_time_tvm = None
    if not args.skip_tvm:
        print("\n--- Running TVM Inference for Benchmark ---")
        inference_time_tvm = run_tvm_inference(
            model_id=args.model_id,
            prompt=args.prompt,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            output_image_prefix="tvm_generated_image",
            tvm_device="cuda" if torch.cuda.is_available() else "cpu",
            tuning_trials=0,
        )

    # Print benchmark results
    print("\n--- Benchmark Results ---")
    if inference_time_pytorch is not None:
        print(f"PyTorch Inference Time: {inference_time_pytorch:.4f} seconds")
    if inference_time_iree is not None:
        print(f"IREE Inference Time   : {inference_time_iree:.4f} seconds")
        if inference_time_pytorch is not None and inference_time_iree > 0:
            speedup = inference_time_pytorch / inference_time_iree
            print(f"IREE Speedup        : {speedup:.2f}x")
    elif inference_time_pytorch is not None:
        print("IREE Inference Time   : FAILED")

    if inference_time_optimum_onnx is not None:
        print(f"Optimum ONNX Time   : {inference_time_optimum_onnx:.4f} seconds")
        if inference_time_pytorch is not None and inference_time_optimum_onnx > 0:
            speedup_onnx = inference_time_pytorch / inference_time_optimum_onnx
            print(f"Optimum ONNX Speedup: {speedup_onnx:.2f}x")
    elif not args.skip_onnx:
        print("Optimum ONNX Time   : FAILED")

    if inference_time_direct_onnx is not None:
        print(f"Direct ONNX Time    : {inference_time_direct_onnx:.4f} seconds")
        if inference_time_pytorch is not None and inference_time_direct_onnx > 0:
            speedup_direct_onnx = inference_time_pytorch / inference_time_direct_onnx
            print(f"Direct ONNX Speedup : {speedup_direct_onnx:.2f}x")
    elif not args.skip_direct_onnx:
        print("Direct ONNX Time    : FAILED / SKIPPED")

    if inference_time_tvm is not None:
        print(f"TVM Time           : {inference_time_tvm:.4f} seconds")
        if inference_time_pytorch is not None and inference_time_tvm > 0:
            speedup_tvm = inference_time_pytorch / inference_time_tvm
            print(f"TVM Speedup        : {speedup_tvm:.2f}x")
    elif not args.skip_tvm:
        print("TVM Time           : FAILED / SKIPPED")

    print("\nPipeline finished.")
