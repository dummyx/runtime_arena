import torch


import os

from diffusers import PNDMScheduler
from transformers import CLIPTokenizer
from optimum.onnxruntime import ORTStableDiffusionPipeline

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel


from PIL import Image
import time
import numpy as np 
import iree.runtime as ireert 
from iree.runtime import FunctionInvoker
import logging 
import gc 
import argparse 
import onnxruntime as ort 

from tqdm.auto import tqdm


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


def run_iree_inference(model_id, vmfb_dir, prompt, batch_size, height, width, num_inference_steps, guidance_scale, output_image_prefix, iree_target_backend="llvm-cpu", iree_device="local-task"):
    """Runs the diffusion inference pipeline using compiled IREE VMFB modules."""
    logger.info("Starting IREE inference...")
    logger.info(f"Using IREE target backend for models: {iree_target_backend}")

    
    logger.info(f"Loading scheduler and tokenizer from {model_id}...")
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    
    vae_scaling_factor = 0.18215
    unet_in_channels = 4
    logger.info("Scheduler and tokenizer loaded.")
    logger.info(f"Using VAE scaling factor: {vae_scaling_factor}, UNet in_channels: {unet_in_channels}")

    
    unet_path = os.path.join(vmfb_dir, f'unet_{iree_target_backend}.vmfb')
    text_encoder_path = os.path.join(vmfb_dir, f'text_encoder_{iree_target_backend}.vmfb')
    vae_decoder_path = os.path.join(vmfb_dir, f'vae_decoder_{iree_target_backend}.vmfb')

    logger.info(f"Looking for VMFB modules in: {vmfb_dir}")
    logger.info(f" - UNet: {os.path.basename(unet_path)}")
    logger.info(f" - Text Encoder: {os.path.basename(text_encoder_path)}")
    logger.info(f" - VAE Decoder: {os.path.basename(vae_decoder_path)}")

    if not all(os.path.exists(p) for p in [unet_path, text_encoder_path, vae_decoder_path]):
        logger.error(f"One or more VMFB files not found in {vmfb_dir} for target '{iree_target_backend}'. Compile ONNX models first with the correct target.")
        return None

    try:
        
        
        logger.info(f"Initializing IREE runtime with driver: {iree_target_backend}")
        config = ireert.Config(driver_name=iree_device)
        unet_ctx = ireert.SystemContext(config=config)
        text_encoder_ctx = ireert.SystemContext(config=config)
        vae_decoder_ctx = ireert.SystemContext(config=config)

        device = ireert.get_first_device()

        
        
        

        logger.info(f"Loading VMFB module: {unet_path}")
        with open(unet_path, 'rb') as f:
            unet_vmfb = f.read()
        logger.info(f"Loading VMFB module: {text_encoder_path}")
        with open(text_encoder_path, 'rb') as f:
            text_encoder_vmfb = f.read()
        logger.info(f"Loading VMFB module: {vae_decoder_path}")
        with open(vae_decoder_path, 'rb') as f:
            vae_decoder_vmfb = f.read()

        logger.info("Creating IREE VM Modules...")
        unet_module = ireert.VmModule.from_flatbuffer(unet_ctx.instance, unet_vmfb)
        text_encoder_module = ireert.VmModule.from_flatbuffer(text_encoder_ctx.instance, text_encoder_vmfb)
        vae_decoder_module = ireert.VmModule.from_flatbuffer(vae_decoder_ctx.instance, vae_decoder_vmfb)
        
        unet_ctx.add_vm_module(unet_module)
        text_encoder_ctx.add_vm_module(text_encoder_module)
        vae_decoder_ctx.add_vm_module(vae_decoder_module)

        logger.info("Looking up IREE function entry points (assuming 'main')...")
        iree_unet = unet_module.lookup_function("main_graph")
        iree_text_encoder = text_encoder_module.lookup_function("main_graph")
        iree_vae_decoder = vae_decoder_module.lookup_function("main_graph")

        unet_invoker = FunctionInvoker(unet_ctx.vm_context, device, iree_unet)
        text_encoder_invoker = FunctionInvoker(text_encoder_ctx.vm_context, device, iree_text_encoder)
        vae_decoder_invoker = FunctionInvoker(vae_decoder_ctx.vm_context, device, iree_vae_decoder)

    except Exception as e:
        logger.error(f"ERROR loading IREE modules or functions: {e}")
        raise e
    logger.info("IREE modules loaded and functions looked up successfully.")

    
    prompts = [prompt] * batch_size
    start_time = time.time()

    
    logger.info("Encoding prompts using IREE Text Encoder...")
    text_input = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_input_ids_np = text_input.input_ids.numpy()

    try:
        
        
        text_embeddings_device, _ = text_encoder_invoker(text_input_ids_np)
        text_embeddings = text_embeddings_device.to_host()
        
        
    except Exception as e:
        logger.error(f"ERROR during IREE Text Encoder inference: {e}")
        raise e

    
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_input_ids_np = uncond_input.input_ids.numpy()

    try:
        
        uncond_embeddings_device, _ = text_encoder_invoker(uncond_input_ids_np)
        uncond_embeddings = uncond_embeddings_device.to_host()
    except Exception as e:
        logger.error(f"ERROR during IREE Text Encoder inference (uncond): {e}")
        raise e

    
    text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
    logger.info("Prompts encoded.")

    
    logger.info("Preparing initial latents...")
    latents_shape = (batch_size, unet_in_channels, height // 8, width // 8)
    latents = np.random.randn(*latents_shape).astype(np.float32)
    
    latents = latents * scheduler.init_noise_sigma
    logger.info("Latents prepared.")

    
    logger.info(f"Starting denoising loop for {num_inference_steps} steps...")
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.numpy() 

    
    
    
    text_embeddings_torch = torch.from_numpy(text_embeddings) 

    
    latents_torch = torch.from_numpy(latents)

    for t in tqdm(scheduler.timesteps):
        
        
        
        
        
        latent_model_input_torch = torch.cat([latents_torch] * 2)
        latent_model_input_scaled_torch = scheduler.scale_model_input(latent_model_input_torch, timestep=t)
        
        latent_model_input_scaled_np = latent_model_input_scaled_torch.numpy()

        
        
        
        timestep_np = np.array(t, dtype=np.float32)

        
        try:
            
            start_invoke = time.time()
            noise_pred_device1, noise_pred_device2 = unet_invoker(latent_model_input_scaled_np,
                                        timestep_np,
                                        text_embeddings 
                                        )
            end_invoke = time.time()
            logger.info(f"IREE UNet invoke time: {end_invoke - start_invoke:.4f} seconds")
            noise_pred = torch.cat([torch.from_numpy(noise_pred_device1), torch.from_numpy(noise_pred_device2)], dim=0)
        except Exception as e:
            logger.error(f"ERROR during IREE UNet inference (step {i}): {e}")
            raise e

        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        
        
        latents_output = scheduler.step(noise_pred, t, latents_torch)
        latents_torch = latents_output.prev_sample

        
        latents = latents_torch.numpy()

        logger.info(f"  Step {t}/{num_inference_steps} completed.")
    logger.info("Denoising loop finished.")

    
    logger.info("Decoding latents using IREE VAE Decoder...")
    
    latents = 1 / vae_scaling_factor * latents
    try:
        
        image_device = vae_decoder_invoker(latents)
        image = image_device.to_host()
    except Exception as e:
        logger.error(f"ERROR during IREE VAE Decoder inference: {e}")
        raise e
    logger.info("Latents decoded.")

    
    end_time = time.time()
    inference_time = end_time - start_time
    logger.info(f"IREE Inference core time: {inference_time:.4f} seconds")

    
    logger.info("Post-processing image...")
    image = (image / 2 + 0.5).clip(0, 1) 
    
    
    
    
    images = (image * 255).round().astype("uint8")

    """
    saved_files = []
    for i, img_np in enumerate(images):
        img = Image.fromarray(img_np)
        filename = f"{output_image_prefix}_{i}.png"
        img.save(filename)
        saved_files.append(filename)
    logger.info(f"Generated images saved as: {', '.join(saved_files)}")
    """
    return inference_time



def run_pytorch_inference(model_id, prompt, batch_size, height, width, num_inference_steps, guidance_scale, output_image_prefix):
    """Runs the diffusion inference pipeline using standard PyTorch Diffusers models."""
    logger.info("Starting PyTorch inference benchmark...")

    
    logger.info(f"Loading PyTorch models from {model_id} for benchmark...")
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder_pt = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet_pt = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    vae_pt = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    logger.info("PyTorch models loaded.")

    vae_scaling_factor = vae_pt.config.scaling_factor
    unet_in_channels = unet_pt.config.in_channels

    prompts = [prompt] * batch_size
    start_time = time.time()

    
    logger.info("Encoding prompts (PyTorch)...")
    text_input = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_input_ids = text_input.input_ids

    with torch.no_grad():
        text_outputs = text_encoder_pt(text_input_ids)
        text_embeddings = text_outputs.last_hidden_state

    
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_input_ids = uncond_input.input_ids
    with torch.no_grad():
        uncond_outputs = text_encoder_pt(uncond_input_ids)
        uncond_embeddings = uncond_outputs.last_hidden_state

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    logger.info("Prompts encoded (PyTorch).")

    
    logger.info("Preparing initial latents (PyTorch)...")
    latents = torch.randn(
        (batch_size, unet_in_channels, height // 8, width // 8),
        dtype=text_embeddings.dtype,
    )
    latents = latents * scheduler.init_noise_sigma
    logger.info("Latents prepared (PyTorch).")

    
    logger.info(f"Starting denoising loop for {num_inference_steps} steps (PyTorch)...")
    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        
        
        
        
        
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        
        with torch.no_grad():
            noise_pred_output = unet_pt(latent_model_input, timestep=t, encoder_hidden_states=text_embeddings)
            noise_pred = noise_pred_output.sample 

        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    logger.info("Denoising loop finished (PyTorch).")

    
    logger.info("Decoding latents (PyTorch)...")
    latents = 1 / vae_scaling_factor * latents
    with torch.no_grad():
        image = vae_pt.decode(latents).sample
    logger.info("Latents decoded (PyTorch).")

    
    end_time = time.time()
    inference_time = end_time - start_time
    logger.info(f"PyTorch Inference core time: {inference_time:.4f} seconds")

    
    logger.info("Post-processing image (PyTorch)...")
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in images]

    saved_files = []
    for i, img in enumerate(pil_images):
        filename = f"{output_image_prefix}_{i}.png"
        img.save(filename)
        saved_files.append(filename)
    logger.info(f"Generated PyTorch images saved as: {', '.join(saved_files)}")

    
    del text_encoder_pt, unet_pt, vae_pt, scheduler, tokenizer, text_embeddings, latents, noise_pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleaned up PyTorch benchmark resources.")

    return inference_time



def run_optimum_onnx_inference(model_id, prompt, batch_size, height, width, num_inference_steps, guidance_scale, output_image_prefix):
    """Runs the diffusion inference pipeline using Optimum ONNX Runtime."""
    logger.info("Starting Optimum ONNX Runtime inference...")
    
    
    
    provider = "CPUExecutionProvider" 
    logger.info(f"Using ONNX Runtime provider: {provider}")

    
    start_load_time = time.time()
    logger.info(f"Loading Optimum ONNX pipeline for {model_id}...")
    try:
        
        ort_pipe = ORTStableDiffusionPipeline.from_pretrained(model_id, provider=provider, export=True) 
        
    except Exception as e:
        logger.error(f"Failed to load Optimum ONNX pipeline: {e}")
        logger.error("Ensure you have 'onnxruntime' or 'onnxruntime-gpu' installed and potentially run with export=True once.")
        return None
    end_load_time = time.time()
    logger.info(f"Optimum ONNX Pipeline loaded in {end_load_time - start_load_time:.2f} seconds.")

    
    
    

    
    logger.info(f"Running Optimum ONNX inference for {num_inference_steps} steps...")
    start_time = time.time()
    
    
    
    images = ort_pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
        
    ).images

    
    end_time = time.time()
    inference_time = end_time - start_time
    logger.info(f"Optimum ONNX Inference core time: {inference_time:.4f} seconds")

    
    
    logger.info("Saving generated Optimum ONNX image(s)...")
    saved_files = []
    
    for i, img in enumerate(images):
        filename = f"{output_image_prefix}_onnx_{i}.png"
        img.save(filename)
        saved_files.append(filename)
    logger.info(f"Generated Optimum ONNX images saved as: {', '.join(saved_files)}")

    
    del ort_pipe, images
    gc.collect()
    
    logger.info("Cleaned up Optimum ONNX benchmark resources.")

    return inference_time



def run_direct_onnx_inference(model_dir, model_id, prompt, batch_size, height, width, num_inference_steps, guidance_scale, output_image_prefix):
    """Runs the diffusion inference pipeline using direct ONNX Runtime."""
    logger.info("Starting Direct ONNX Runtime inference...")
    start_time = time.time()

    
    logger.info(f"Loading ONNX models from {model_dir}...")
    try:
        sess_options = ort.SessionOptions()
        
        
        
        provider = 'CPUExecutionProvider'
        logger.info(f"Using ONNX Runtime provider: {provider}")

        text_encoder_session = ort.InferenceSession(os.path.join(model_dir, "text_encoder.onnx"), sess_options=sess_options, providers=[provider])
        unet_session = ort.InferenceSession(os.path.join(model_dir, "unet.onnx"), sess_options=sess_options, providers=[provider])
        vae_decoder_session = ort.InferenceSession(os.path.join(model_dir, "vae_decoder.onnx"), sess_options=sess_options, providers=[provider])
        logger.info("ONNX models loaded.")
    except Exception as e:
        logger.error(f"Failed to load ONNX models: {e}")
        logger.error(f"Ensure ONNX models exist in separate directories within {model_dir} (e.g., {model_dir}/text_encoder/model.onnx)")
        return None

    
    
    
    
    logger.info(f"Loading scheduler and tokenizer from {model_id}...")
    try:
        scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        vae_scaling_factor = 0.18215 
        logger.info("Scheduler and tokenizer loaded.")
    except Exception as e:
        logger.error(f"Failed to load scheduler/tokenizer from {model_id}: {e}")
        return None

    
    logger.info("Preparing inputs...")
    
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np")
    input_ids = text_input.input_ids

    
    logger.info("Running text encoder...")
    encoder_hidden_states = text_encoder_session.run(None, {'input_ids': input_ids.astype(np.int64)})[0] 

    
    uncond_input = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np")
    uncond_input_ids = uncond_input.input_ids
    uncond_embeddings = text_encoder_session.run(None, {'input_ids': uncond_input_ids.astype(np.int64)})[0]

    
    text_embeddings = np.concatenate([uncond_embeddings] * batch_size + [encoder_hidden_states] * batch_size)

    
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.numpy()

    
    latents_shape = (batch_size, 4, height // 8, width // 8) 
    latents = np.random.randn(*latents_shape).astype(np.float32)
    latents = latents * scheduler.init_noise_sigma

    logger.info("Starting diffusion loop...")
    
    loop_start_time = time.time()
    for i, t in enumerate(tqdm(timesteps)):
        
        latent_model_input = np.concatenate([latents] * 2) 
        latent_model_input = scheduler.scale_model_input(torch.from_numpy(latent_model_input), t).numpy() 

        
        timestep_input = np.array([t], dtype=np.int64)

        
        
        
        unet_inputs = {
            'sample': latent_model_input.astype(np.float32),
            'timestep': timestep_input.astype(np.float32),
            'encoder_hidden_states': text_embeddings.astype(np.float32) 
        }
        noise_pred = unet_session.run(None, unet_inputs)[0] 

        
        noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        
        latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()

    loop_end_time = time.time()
    logger.info(f"Diffusion loop finished in {loop_end_time - loop_start_time:.4f} seconds.")

    
    logger.info("Decoding latents with VAE...")
    
    latents = 1 / vae_scaling_factor * latents
    
    vae_inputs = {'latent_sample': latents.astype(np.float32)}
    image = vae_decoder_session.run(None, vae_inputs)[0] 

    
    image = (image / 2 + 0.5).clip(0, 1)
    image = (image * 255).round().astype("uint8")
    
    image = np.transpose(image, (0, 2, 3, 1))

    pil_images = [Image.fromarray(img) for img in image]

    for i, img in enumerate(pil_images):
        output_path = f"{output_image_prefix}_{i+1}.png"
        img.save(output_path)
        logger.info(f"Saved image to {output_path}")

    gc.collect() 

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Direct ONNX Runtime inference finished in {total_time:.4f} seconds.")
    return total_time


if __name__ == "__main__":
    print("--- Stable Diffusion IREE Pipeline ---")

    parser = argparse.ArgumentParser(description="Benchmark Stable Diffusion Inference (PyTorch, ONNX Runtime, IREE).")
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

    args = parser.parse_args()

    
    model_dir = args.model_dir
    iree_target = args.iree_target
    iree_device = args.iree_device

    
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
            
            
            
            model_id=MODEL_ID, 
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
                model_id=MODEL_ID, 
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
    elif RUN_BENCHMARK:
        print("Optimum ONNX Time   : FAILED")

    if inference_time_direct_onnx is not None:
        print(f"Direct ONNX Time    : {inference_time_direct_onnx:.4f} seconds")
        if inference_time_pytorch is not None and inference_time_direct_onnx > 0:
            
            speedup_direct_onnx = inference_time_pytorch / inference_time_direct_onnx
            print(f"Direct ONNX Speedup : {speedup_direct_onnx:.2f}x")
    elif not args.skip_direct_onnx:
        print("Direct ONNX Time    : FAILED / SKIPPED")

    print("\nPipeline finished.")
