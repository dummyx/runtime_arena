import torch
import torch.onnx
import os
import sys
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
import logging
import subprocess
import shutil


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pytorch_models(model_id):
    """Loads necessary PyTorch models from Hugging Face Hub.
    
    Args:
        model_id: Hugging Face model ID
        
    Returns:
        tuple: (tokenizer, text_encoder, unet, vae)
        
    Raises:
        Exception: If models cannot be loaded
    """
    try:
        logger.info(f"Loading PyTorch models from {model_id}...")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        logger.info("PyTorch models loaded.")
        return tokenizer, text_encoder, unet, vae
    except Exception as e:
        logger.error(f"Error loading PyTorch models: {e}")
        raise



def export_models(model_id, batch_size, height, width, output_dir, opset=17, skip_existing=True, compile_iree=True, iree_target_backend='cuda'):
    """Exports UNet, Text Encoder, and VAE Decoder to ONNX format and optionally compiles to IREE VMFB.
    
    Args:
        model_id: Hugging Face model ID
        batch_size: Batch size for inference
        height: Image height
        width: Image width
        output_dir: Directory to save models
        opset: ONNX opset version
        skip_existing: Skip export if files exist
        compile_iree: Whether to compile to IREE VMFB
        iree_target_backend: IREE target backend
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    os.makedirs(output_dir, exist_ok=True)

    
    unet_onnx_path = os.path.join(output_dir, 'unet.onnx')
    text_encoder_onnx_path = os.path.join(output_dir, 'text_encoder.onnx')
    vae_decoder_onnx_path = os.path.join(output_dir, 'vae_decoder.onnx')
    unet_vmfb_path = os.path.join(output_dir, f'unet_{iree_target_backend}.vmfb')
    text_encoder_vmfb_path = os.path.join(output_dir, f'text_encoder_{iree_target_backend}.vmfb')
    vae_decoder_vmfb_path = os.path.join(output_dir, f'vae_decoder_{iree_target_backend}.vmfb')

    
    all_targets_exist = (
        os.path.exists(unet_onnx_path) and
        os.path.exists(text_encoder_onnx_path) and
        os.path.exists(vae_decoder_onnx_path) and
        (not compile_iree or (os.path.exists(unet_vmfb_path) and os.path.exists(text_encoder_vmfb_path) and os.path.exists(vae_decoder_vmfb_path)))
    )

    if skip_existing and all_targets_exist:
        logger.info(f"All target ONNX and IREE VMFB files already exist in {output_dir}. Skipping export and compilation.")
        return True

    
    try:
        tokenizer, text_encoder, unet, vae = load_pytorch_models(model_id)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

    
    export_needed = (
        not os.path.exists(unet_onnx_path) or
        not os.path.exists(text_encoder_onnx_path) or
        not os.path.exists(vae_decoder_onnx_path)
    )
    if export_needed:
        try:
            logger.info("Generating dummy inputs for ONNX export...")
            latent_channels = unet.config.in_channels
            embedding_dim = text_encoder.config.hidden_size
            model_max_length = tokenizer.model_max_length
            vae_latent_channels = vae.config.latent_channels
            image_height, image_width = height, width

            unet_inference_batch_size = batch_size * 2 
            dummy_sample = torch.randn(unet_inference_batch_size, latent_channels, image_height // 8, image_width // 8, dtype=torch.float32)
            dummy_timestep = torch.tensor(999, dtype=torch.float32)
            dummy_encoder_hidden_states = torch.randn(unet_inference_batch_size, model_max_length, embedding_dim, dtype=torch.float32)

            dummy_text_input_ids = tokenizer(["dummy prompt"] * batch_size, padding="max_length", max_length=model_max_length, truncation=True, return_tensors="pt").input_ids

            dummy_latents_for_vae = torch.randn(batch_size, vae_latent_channels - 1, image_height // 8, image_width // 8, dtype=torch.float32)
            logger.info("Dummy inputs generated.")
        except Exception as e:
            logger.error(f"Error generating dummy inputs: {e}")
            return False

    
    unet_exported = False
    if not skip_existing or not os.path.exists(unet_onnx_path):
        try:
            logger.info(f"Exporting UNet to {unet_onnx_path} with opset {opset}...")
            unet.eval()
            
            
            dynamic_axes = {
                "sample": {0: "batch"},
                "timestep": {},
                "encoder_hidden_states": {0: "batch"},
                "out_sample": {0: "batch"},
            }
            
            
            with torch.no_grad():
                torch.onnx.export(
                    unet,
                    (dummy_sample, dummy_timestep, dummy_encoder_hidden_states),
                    unet_onnx_path,
                    input_names=["sample", "timestep", "encoder_hidden_states"],
                    output_names=["out_sample"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset,
                )
            logger.info(f"UNet exported to {unet_onnx_path}.")
            unet_exported = True
        except Exception as e:
            logger.error(f"Error exporting UNet: {e}")

    
    text_encoder_exported = False
    if not skip_existing or not os.path.exists(text_encoder_onnx_path):
        try:
            logger.info(f"Exporting Text Encoder to {text_encoder_onnx_path} with opset {opset}...")
            text_encoder.eval()
            
            
            dynamic_axes = {
                "input_ids": {0: "batch"},
                "last_hidden_state": {0: "batch"},
            }
            
            
            with torch.no_grad():
                torch.onnx.export(
                    text_encoder,
                    dummy_text_input_ids,
                    text_encoder_onnx_path,
                    input_names=["input_ids"],
                    output_names=["last_hidden_state"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset,
                )
            logger.info(f"Text Encoder exported to {text_encoder_onnx_path}.")
            text_encoder_exported = True
        except Exception as e:
            logger.error(f"Error exporting Text Encoder: {e}")

    

    vae_decoder_exported = False
    if not skip_existing or not os.path.exists(vae_decoder_onnx_path):
        try:
            logger.info(f"Exporting VAE Decoder to {vae_decoder_onnx_path} with opset {opset}...")
            
            
            dynamic_axes = {
                "latents": {0: "batch"},
                "images": {0: "batch"},
            }
            
            
            with torch.no_grad():
                torch.onnx.export(
                    vae,
                    dummy_latents_for_vae,
                    vae_decoder_onnx_path,
                    input_names=["latents"],
                    output_names=["images"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset,
                )
            logger.info(f"VAE Decoder exported to {vae_decoder_onnx_path}.")
            vae_decoder_exported = True
        except Exception as e:
            logger.error(f"Error exporting VAE Decoder: {e}")

    
    if compile_iree:
        logger.info("\n--- Starting IREE Compilation ---")
        
        
        compilation_results = {}
        
        
        compilation_jobs = [
            {
                "name": "unet",
                "onnx_path": unet_onnx_path,
                "mlir_path": os.path.join(output_dir, "unet.mlir"),
                "vmfb_path": unet_vmfb_path,
                "exported": unet_exported or (skip_existing and os.path.exists(unet_onnx_path))
            },
            {
                "name": "text_encoder",
                "onnx_path": text_encoder_onnx_path,
                "mlir_path": os.path.join(output_dir, "text_encoder.mlir"),
                "vmfb_path": text_encoder_vmfb_path,
                "exported": text_encoder_exported or (skip_existing and os.path.exists(text_encoder_onnx_path))
            },
            {
                "name": "vae_decoder",
                "onnx_path": vae_decoder_onnx_path,
                "mlir_path": os.path.join(output_dir, "vae_decoder.mlir"),
                "vmfb_path": vae_decoder_vmfb_path,
                "exported": vae_decoder_exported or (skip_existing and os.path.exists(vae_decoder_onnx_path))
            }
        ]
        
        
        for job in compilation_jobs:
            name = job["name"]
            onnx_path = job["onnx_path"]
            mlir_path = job["mlir_path"]
            vmfb_path = job["vmfb_path"]
            exported = job["exported"]
            
            compilation_results[name] = {
                "import": False,
                "compile": False
            }
            
            if not exported:
                logger.warning(f"Skipping {name} compilation as ONNX export failed or was skipped.")
                continue
                
            if skip_existing and os.path.exists(vmfb_path):
                logger.info(f"{name} VMFB already exists at {vmfb_path}. Skipping compilation.")
                compilation_results[name]["import"] = True
                compilation_results[name]["compile"] = True
                continue

            
            if not os.path.exists(mlir_path) or not skip_existing:
                logger.info(f"Importing {name} ONNX from {onnx_path} to MLIR...")
                import_command = [
                    'uv', 'run',
                    'iree-import-onnx',
                    onnx_path, 
                    '--opset-version', '17',
                    '-o', mlir_path 
                ]
                try:
                    import_result = subprocess.run(import_command, check=True, capture_output=True, text=True)
                    logger.info(f"Successfully imported {name} ONNX to MLIR.")
                    logger.debug(f"IREE Import Output:\n{import_result.stdout}")
                    if import_result.stderr:
                        logger.debug(f"IREE Import stderr:\n{import_result.stderr}")
                    compilation_results[name]["import"] = True
                except subprocess.CalledProcessError as e:
                    logger.error(f"ERROR importing {name} ONNX with IREE: {e}") 
                    logger.error(f"Command: {' '.join(e.cmd)}")
                    logger.error(f"Stderr: {e.stderr}")
                    continue 
                except FileNotFoundError:
                    logger.error(f"ERROR: 'iree-import-onnx' command not found. Make sure IREE is installed and in your PATH.")
                    continue 
            else:
                 logger.info(f"MLIR file {mlir_path} already exists. Skipping import for {name}.")
                 compilation_results[name]["import"] = True
            
            
            
            if not os.path.exists(mlir_path):
                 logger.error(f"Cannot compile {name}: MLIR file {mlir_path} not found (import might have failed).")
                 continue

            
            logger.info(f"Compiling {name} MLIR from {mlir_path} to {vmfb_path} for target {iree_target_backend}...")
            compile_command = [
                'uv', 'run',
                'iree-compile',
                '--iree-hal-target-device=cuda',
                
                
                '--iree-opt-level=O2',
                mlir_path, 
                '-o',
                vmfb_path
            ]
            try:
                compile_result = subprocess.run(compile_command, check=True, capture_output=True, text=True)
                logger.info(f"Successfully compiled {name} to {vmfb_path}.")
                logger.debug(f"IREE Compile Output:\n{compile_result.stdout}")
                if compile_result.stderr:
                    logger.debug(f"IREE Compile stderr:\n{compile_result.stderr}")
                compilation_results[name]["compile"] = True
            except subprocess.CalledProcessError as e:
                logger.error(f"ERROR compiling {name} MLIR with IREE: {e}")
                logger.error(f"Command: {' '.join(e.cmd)}")
                logger.error(f"Stderr: {e.stderr}")
            except FileNotFoundError:
                 logger.error(f"ERROR: 'iree-compile' command not found. Make sure IREE is installed and in your PATH.")
        
        
        logger.info("\n=== IREE Compilation Summary ===")
        all_successful = True
        for name, result in compilation_results.items():
            status = "✓ Success" if result["import"] and result["compile"] else "❌ Failed"
            logger.info(f"{name}: {status}")
            if not (result["import"] and result["compile"]):
                all_successful = False
        
        if not all_successful and not skip_existing:
            logger.warning("Some models failed to compile. Check logs for details.")
        
        return all_successful
    else:
        
        all_exported = unet_exported and text_encoder_exported and vae_decoder_exported
        if not all_exported and not skip_existing:
            logger.warning("Some models failed to export to ONNX. Check logs for details.")
        
        return all_exported


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
    parser.add_argument("--iree_target", type=str, default="cuda", help="IREE target backend (e.g., cuda, vulkan, cuda).")

    args = parser.parse_args()

    success = export_models(
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
    
    sys.exit(0 if success else 1)
