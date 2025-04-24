import torch
import torch.onnx
import os
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
def load_pytorch_models(model_id):
    """Loads necessary PyTorch models from Hugging Face Hub."""
    logger.info(f"Loading PyTorch models from {model_id}...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    logger.info("PyTorch models loaded.")
    return scheduler, tokenizer, text_encoder, unet, vae

# --- Export Function ---
def export_models(model_id, batch_size, height, width, output_dir, opset=17, skip_existing=True, compile_iree=True, iree_target_backend='llvm-cpu'):
    """Exports UNet, Text Encoder, and VAE Decoder to ONNX format and optionally compiles to IREE VMFB."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Define Paths --- 
    unet_onnx_path = os.path.join(output_dir, 'unet.onnx')
    text_encoder_onnx_path = os.path.join(output_dir, 'text_encoder.onnx')
    vae_decoder_onnx_path = os.path.join(output_dir, 'vae_decoder.onnx')
    unet_vmfb_path = os.path.join(output_dir, f'unet_{iree_target_backend}.vmfb')
    text_encoder_vmfb_path = os.path.join(output_dir, f'text_encoder_{iree_target_backend}.vmfb')
    vae_decoder_vmfb_path = os.path.join(output_dir, f'vae_decoder_{iree_target_backend}.vmfb')

    # Check if all final target files exist if skip_existing is True
    all_targets_exist = (
        os.path.exists(unet_onnx_path) and
        os.path.exists(text_encoder_onnx_path) and
        os.path.exists(vae_decoder_onnx_path) and
        (not compile_iree or (os.path.exists(unet_vmfb_path) and os.path.exists(text_encoder_vmfb_path) and os.path.exists(vae_decoder_vmfb_path)))
    )

    if skip_existing and all_targets_exist:
        logger.info(f"All target ONNX and IREE VMFB files already exist in {output_dir}. Skipping export and compilation.")
        return

    # --- Load PyTorch Models (only if needed) ---
    scheduler, tokenizer, text_encoder, unet, vae = load_pytorch_models(model_id)

    # --- Dummy Inputs (only if needed for export) ---
    export_needed = (
        not os.path.exists(unet_onnx_path) or
        not os.path.exists(text_encoder_onnx_path) or
        not os.path.exists(vae_decoder_onnx_path)
    )
    if export_needed:
        logger.info("Generating dummy inputs for ONNX export...")
        latent_channels = unet.config.in_channels
        embedding_dim = text_encoder.config.hidden_size
        model_max_length = tokenizer.model_max_length
        vae_latent_channels = vae.config.latent_channels
        image_height, image_width = height, width

        unet_inference_batch_size = batch_size * 2 # Doubled for CFG
        dummy_sample = torch.randn(unet_inference_batch_size, latent_channels, image_height // 8, image_width // 8, dtype=torch.float32)
        dummy_timestep = torch.tensor(999, dtype=torch.float32)
        dummy_encoder_hidden_states = torch.randn(unet_inference_batch_size, model_max_length, embedding_dim, dtype=torch.float32)

        dummy_text_input_ids = tokenizer(["dummy prompt"] * batch_size, padding="max_length", max_length=model_max_length, truncation=True, return_tensors="pt").input_ids

        dummy_latents_for_vae = torch.randn(batch_size, vae_latent_channels, image_height // 8, image_width // 8, dtype=torch.float32)
        logger.info("Dummy inputs generated.")

    # --- Export UNet to ONNX ---
    if not skip_existing or not os.path.exists(unet_onnx_path):
        logger.info(f"Exporting UNet to {unet_onnx_path} with opset {opset}...")
        unet.eval()
        try:
            example_inputs_unet = (dummy_sample, dummy_timestep, dummy_encoder_hidden_states)
            torch.onnx.export(
                unet,
                example_inputs_unet,
                unet_onnx_path,
                input_names=['sample', 'timestep', 'encoder_hidden_states'],
                output_names=['out_sample'],
                dynamic_axes={
                    'sample': {0: 'batch_size'}, 
                    'encoder_hidden_states': {0: 'batch_size'} 
                },
                opset_version=opset
            )
            logger.info(f"UNet ONNX model saved to {unet_onnx_path}")
        except Exception as e:
            logger.error(f"ERROR exporting UNet to ONNX: {e}")
            raise
    else:
        logger.info(f"UNet ONNX model already exists at {unet_onnx_path}. Skipping export.")

    # --- Export Text Encoder to ONNX ---
    if not skip_existing or not os.path.exists(text_encoder_onnx_path):
        logger.info(f"Exporting Text Encoder to {text_encoder_onnx_path} with opset {opset}...")
        text_encoder.eval()
        try:
            torch.onnx.export(
                text_encoder,
                (dummy_text_input_ids,), 
                text_encoder_onnx_path,
                input_names=['input_ids'],
                output_names=['last_hidden_state'], 
                dynamic_axes={'input_ids': {0: 'batch_size'}}, 
                opset_version=opset
            )
            logger.info(f"Text Encoder ONNX model saved to {text_encoder_onnx_path}")
        except Exception as e:
            logger.error(f"ERROR exporting Text Encoder to ONNX: {e}")
            raise
    else:
        logger.info(f"Text Encoder ONNX model already exists at {text_encoder_onnx_path}. Skipping export.")

    # --- Export VAE Decoder to ONNX ---
    if not skip_existing or not os.path.exists(vae_decoder_onnx_path):
        logger.info(f"Exporting VAE Decoder to {vae_decoder_onnx_path} with opset {opset}...")
        vae.eval()
        class VAEDecodeWrapper(torch.nn.Module):
                def __init__(self, vae_model):
                    super().__init__()
                    self.vae = vae_model
                def forward(self, latents):
                    return self.vae.decode(latents).sample

        vae_wrapper = VAEDecodeWrapper(vae)
        vae_wrapper.eval()

        try:
            torch.onnx.export(
                vae_wrapper,
                (dummy_latents_for_vae,),
                vae_decoder_onnx_path,
                input_names=['latent_sample'],
                output_names=['sample'],
                dynamic_axes={'latent_sample': {0: 'batch_size'}}, 
                opset_version=opset
            )
            logger.info(f"VAE Decoder ONNX model saved to {vae_decoder_onnx_path}")
        except Exception as e:
            logger.error(f"ERROR exporting VAE Decoder to ONNX: {e}")
            raise
    else:
        logger.info(f"VAE Decoder ONNX model already exists at {vae_decoder_onnx_path}. Skipping export.")

    logger.info("ONNX export process completed (or skipped).")

    # --- Compile ONNX to IREE VMFB --- 
    if compile_iree:
        logger.info("Starting IREE import and compilation...")
        models_to_process = {
            "UNet": (unet_onnx_path, unet_vmfb_path),
            "TextEncoder": (text_encoder_onnx_path, text_encoder_vmfb_path),
            "VAEDecoder": (vae_decoder_onnx_path, vae_decoder_vmfb_path)
        }

        for name, (onnx_path, vmfb_path) in models_to_process.items():
            # Define MLIR path
            mlir_path = onnx_path.replace(".onnx", ".mlir")

            # Check if final VMFB exists first (most efficient skip)
            if skip_existing and os.path.exists(vmfb_path):
                 logger.info(f"IREE VMFB file {vmfb_path} already exists. Skipping import and compilation for {name}.")
                 continue

            # --- Import ONNX to MLIR --- 
            # Check if ONNX input exists before attempting import
            if not os.path.exists(onnx_path):
                 logger.error(f"Cannot import {name}: ONNX file {onnx_path} not found.")
                 continue

            # Check if MLIR needs to be generated
            if not skip_existing or not os.path.exists(mlir_path):
                logger.info(f"Importing {name} ONNX from {onnx_path} to MLIR at {mlir_path}...")
                import_command = [
                    'uv', 'run',
                    'iree-import-onnx',
                    onnx_path,
                    '--opset-version', '17',
                    '-o',
                    mlir_path
                ]
                try:
                    import_result = subprocess.run(import_command, check=True, capture_output=True, text=True)
                    logger.info(f"Successfully imported {name} to {mlir_path}.")
                    logger.debug(f"IREE Import Output:\n{import_result.stdout}")
                    if import_result.stderr:
                        logger.warning(f"IREE Import stderr:\n{import_result.stderr}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"ERROR importing {name} ONNX to MLIR: {e}")
                    logger.error(f"Command: {' '.join(e.cmd)}")
                    logger.error(f"Stderr: {e.stderr}")
                    continue # Skip compilation if import fails
                except FileNotFoundError:
                    logger.error(f"ERROR: 'iree-import-onnx' command not found. Make sure IREE is installed and in your PATH.")
                    continue # Skip compilation if import command not found
            else:
                 logger.info(f"MLIR file {mlir_path} already exists. Skipping import for {name}.")
            
            # --- Compile MLIR to IREE VMFB ---
            # Check if MLIR input exists before attempting compile (might have failed import)
            if not os.path.exists(mlir_path):
                 logger.error(f"Cannot compile {name}: MLIR file {mlir_path} not found (import might have failed).")
                 continue

            # VMFB existence check already done at the top, so just compile if needed
            logger.info(f"Compiling {name} MLIR from {mlir_path} to {vmfb_path} for target {iree_target_backend}...")
            compile_command = [
                'uv', 'run',
                'iree-compile',
                '--iree-hal-target-device=local',
                '--iree-hal-local-target-device-backends=llvm-cpu',
                '--iree-llvmcpu-target-cpu=host',
                '--iree-opt-level=O1',
                '--iree-llvmcpu-target-cpu-features=host',
                mlir_path, # Use MLIR path as input
                '-o',
                vmfb_path
            ]
            try:
                compile_result = subprocess.run(compile_command, check=True, capture_output=True, text=True)
                logger.info(f"Successfully compiled {name} to {vmfb_path}.")
                logger.debug(f"IREE Compile Output:\n{compile_result.stdout}")
                if compile_result.stderr:
                    logger.warning(f"IREE Compile stderr:\n{compile_result.stderr}")
            except subprocess.CalledProcessError as e:
                logger.error(f"ERROR compiling {name} MLIR with IREE: {e}") # Updated error message
                logger.error(f"Command: {' '.join(e.cmd)}")
                logger.error(f"Stderr: {e.stderr}")
            except FileNotFoundError:
                 logger.error(f"ERROR: 'iree-compile' command not found. Make sure IREE is installed and in your PATH.")
                 # Don't break the loop, other models might compile
        
        logger.info("IREE import and compilation process completed (or skipped).") # Updated log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Stable Diffusion models to ONNX and optionally compile to IREE.")
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Hugging Face model ID.")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save ONNX and VMFB models.") 
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dummy inputs and dynamic axes.")
    parser.add_argument("--height", type=int, default=512, help="Image height for dummy inputs.")
    parser.add_argument("--width", type=int, default=512, help="Image width for dummy inputs.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--skip_existing", action='store_true', help="Skip export/compilation if target files exist.")
    parser.add_argument("--no_iree", action='store_true', help="Do not compile ONNX models to IREE VMFB.")
    parser.add_argument("--iree_target", type=str, default="llvm-cpu", help="IREE target backend (e.g., llvm-cpu, vulkan, cuda).")

    args = parser.parse_args()

    export_models(
        model_id=args.model_id,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        output_dir=args.output_dir,
        opset=args.opset,
        skip_existing=args.skip_existing,
        compile_iree=not args.no_iree,
        iree_target_backend=args.iree_target
    )
