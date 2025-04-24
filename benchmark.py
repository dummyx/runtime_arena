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

IREE_TARGET_BACKEND = "cuda"
IREE_DEVICE = "local-task"


def check_iree_models(model_dir, iree_target):
    """Check if required IREE VMFB models exist.
    
    Args:
        model_dir: Directory containing models
        iree_target: IREE target backend identifier
        
    Returns:
        bool: True if all required models exist, False otherwise
    """
    unet_vmfb_path = os.path.join(model_dir, f'unet_{iree_target}.vmfb')
    text_encoder_vmfb_path = os.path.join(model_dir, f'text_encoder_{iree_target}.vmfb')
    vae_decoder_vmfb_path = os.path.join(model_dir, f'vae_decoder_{iree_target}.vmfb')

    print(f"Checking for IREE VMFB files in '{model_dir}' for target '{iree_target}':")
    print(f" - {os.path.basename(unet_vmfb_path)}")
    print(f" - {os.path.basename(text_encoder_vmfb_path)}")
    print(f" - {os.path.basename(vae_decoder_vmfb_path)}")

    all_exist = all(os.path.exists(p) for p in [unet_vmfb_path, text_encoder_vmfb_path, vae_decoder_vmfb_path])
    
    if not all_exist:
        logger.error(f"ERROR: Compiled VMFB models not found in {model_dir} for target '{iree_target}'.")
        logger.error("Please ensure you have exported/compiled the models using 'export_models.py' with the correct --output_dir and --iree_target.")
    else:
        logger.info(f"Found required IREE VMFB models.")
        
    return all_exist


def check_direct_onnx_models(model_dir):
    """Check if required ONNX models exist for direct ONNX inference.
    
    Args:
        model_dir: Directory containing models
        
    Returns:
        bool: True if all required models exist, False otherwise
    """
    text_encoder_path = os.path.join(model_dir, "text_encoder.onnx")
    unet_path = os.path.join(model_dir, "unet.onnx")
    vae_decoder_path = os.path.join(model_dir, "vae_decoder.onnx")
    
    all_exist = all(os.path.exists(p) for p in [text_encoder_path, unet_path, vae_decoder_path])
    
    if not all_exist:
        logger.warning(f"Direct ONNX models not found in expected locations within {model_dir}. Skipping direct ONNX benchmark.")
        logger.warning(f"Expected: {model_dir}/text_encoder.onnx, {model_dir}/unet.onnx, {model_dir}/vae_decoder.onnx")
        
    return all_exist


def print_benchmark_results(results):
    """Print benchmark results in a formatted way.
    
    Args:
        results: Dictionary containing benchmark results
    """
    print("\n--- Benchmark Results ---")
    
    
    pytorch_time = results.get('pytorch')
    iree_time = results.get('iree')
    optimum_onnx_time = results.get('optimum_onnx')
    direct_onnx_time = results.get('direct_onnx')
    tvm_time = results.get('tvm')
    
    
    if pytorch_time is not None:
        print(f"PyTorch Inference Time: {pytorch_time:.4f} seconds")
    
    
    if iree_time is not None:
        print(f"IREE Inference Time   : {iree_time:.4f} seconds")
        if pytorch_time is not None and iree_time > 0:
            speedup = pytorch_time / iree_time
            print(f"IREE Speedup        : {speedup:.2f}x")
    elif pytorch_time is not None and results.get('skip_iree') is False:
        print("IREE Inference Time   : FAILED")

    
    if optimum_onnx_time is not None:
        print(f"Optimum ONNX Time   : {optimum_onnx_time:.4f} seconds")
        if pytorch_time is not None and optimum_onnx_time > 0:
            speedup_onnx = pytorch_time / optimum_onnx_time
            print(f"Optimum ONNX Speedup: {speedup_onnx:.2f}x")
    elif results.get('skip_onnx') is False:
        print("Optimum ONNX Time   : FAILED")

    
    if direct_onnx_time is not None:
        print(f"Direct ONNX Time    : {direct_onnx_time:.4f} seconds")
        if pytorch_time is not None and direct_onnx_time > 0:
            speedup_direct_onnx = pytorch_time / direct_onnx_time
            print(f"Direct ONNX Speedup : {speedup_direct_onnx:.2f}x")
    elif results.get('skip_direct_onnx') is False:
        print("Direct ONNX Time    : FAILED / SKIPPED")

    
    if tvm_time is not None:
        print(f"TVM Time           : {tvm_time:.4f} seconds")
        if pytorch_time is not None and tvm_time > 0:
            speedup_tvm = pytorch_time / tvm_time
            print(f"TVM Speedup        : {speedup_tvm:.2f}x")
    elif results.get('skip_tvm') is False:
        print("TVM Time           : FAILED / SKIPPED")

    print("\nPipeline finished.")


def run_benchmarks(args):
    """Run benchmarks based on provided arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        int: 0 on success, non-zero on failure
    """
    print("--- Stable Diffusion IREE Pipeline ---")

    model_dir = args.model_dir
    iree_target = args.iree_target
    iree_device = args.iree_device

    
    results = {
        'skip_pytorch': args.skip_pytorch,
        'skip_onnx': args.skip_onnx,
        'skip_iree': args.skip_iree,
        'skip_direct_onnx': args.skip_direct_onnx,
        'skip_tvm': args.skip_tvm,
        'pytorch': None,
        'iree': None,
        'optimum_onnx': None,
        'direct_onnx': None,
        'tvm': None
    }

    
    if not args.skip_iree:
        if not check_iree_models(model_dir, iree_target):
            logger.warning("Skipping IREE benchmark due to missing models")
            results['skip_iree'] = True

    
    if not args.skip_iree and not results.get('skip_iree', False):
        try:
            results['iree'] = run_iree_inference(
                model_id=args.model_id, 
                vmfb_dir=model_dir, 
                prompt=args.prompt, 
                batch_size=args.batch_size, 
                height=args.height, 
                width=args.width, 
                num_inference_steps=args.steps, 
                guidance_scale=args.guidance, 
                output_image_prefix=OUTPUT_IMAGE_PREFIX, 
                iree_target_backend=iree_target, 
                iree_device=iree_device
            )
        except Exception as e:
            logger.error(f"IREE inference failed: {e}")

    if not args.skip_pytorch:
        try:
            print("\n--- Running PyTorch Inference for Benchmark ---")
            results['pytorch'] = run_pytorch_inference(
                model_id=args.model_id, 
                prompt=args.prompt, 
                batch_size=args.batch_size, 
                height=args.height, 
                width=args.width, 
                num_inference_steps=args.steps, 
                guidance_scale=args.guidance, 
                output_image_prefix=PYTORCH_OUTPUT_IMAGE_PREFIX
            )
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")

    if not args.skip_onnx:
        try:
            print("\n--- Running Optimum ONNX Runtime Inference for Benchmark ---")
            results['optimum_onnx'] = run_optimum_onnx_inference(
                model_id=args.model_id, 
                prompt=args.prompt, 
                batch_size=args.batch_size, 
                height=args.height, 
                width=args.width, 
                num_inference_steps=args.steps, 
                guidance_scale=args.guidance, 
                output_image_prefix="optimum_generated_image" 
            )
        except Exception as e:
            logger.error(f"Optimum ONNX inference failed: {e}")

    if not args.skip_direct_onnx:
        print("\n--- Running Direct ONNX Runtime Inference for Benchmark ---")
        if check_direct_onnx_models(model_dir):
            try:
                results['direct_onnx'] = run_direct_onnx_inference(
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
            except Exception as e:
                logger.error(f"Direct ONNX inference failed: {e}")

    if not args.skip_tvm:
        try:
            print("\n--- Running TVM Inference for Benchmark ---")
            results['tvm'] = run_tvm_inference(
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
        except Exception as e:
            logger.error(f"TVM inference failed: {e}")

    
    print_benchmark_results(results)
    
    
    successful_benchmarks = sum(1 for v in [results['pytorch'], results['iree'], 
                                           results['optimum_onnx'], results['direct_onnx'], 
                                           results['tvm']] 
                              if v is not None)
    return 0 if successful_benchmarks > 0 else 1


if __name__ == "__main__":

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
    
    
    exit_code = run_benchmarks(args)
    exit(exit_code)
