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


import torch
from torch.export import export
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.frontend import detach_params
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
import numpy as np
from tqdm.auto import tqdm
from time import time
from PIL import Image

logger = logging.getLogger(__name__)


class TVMStableDiffusionPipeline:
    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device="cuda", target_str="cuda", tuning_trials=80):
        self.device = tvm.cuda(0) if device == "cuda" else tvm.cpu(0) 
        self.target = tvm.target.Target(target_str)
        self.tuning_trials = tuning_trials
        self.torch_device = torch.device(device)

        print("Loading models...")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        
        self.text_encoder_pt = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", use_safetensors=True).to(self.torch_device)
        self.unet_pt = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_safetensors=True).to(self.torch_device)
        self.vae_pt = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True).to(self.torch_device)

        
        self.text_encoder_vm, self.text_encoder_params = None, None
        self.unet_vm, self.unet_params = None, None
        self.vae_vm, self.vae_params = None, None

        
        print("Compiling models with TVM...")
        self._compile_text_encoder()
        self._compile_unet()
        self._compile_vae()
        print("TVM compilation complete.")

    def _compile_model(self, pt_model, example_inputs, model_name):
        print(f"  Compiling {model_name}...")
        start_time = time()

        print(f"    Exporting {model_name}...")
        
        pt_model.eval()
        exported_program = export(pt_model, example_inputs)

        print(f"    Converting {model_name} to Relax...")
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        mod, params = detach_params(mod)

        print(f"    Applying static_shape_tuning to {model_name}...")
        
        if self.tuning_trials > 0:
            pipeline = relax.get_pipeline("static_shape_tuning", target=self.target, total_trials=self.tuning_trials)
            try:
                mod = pipeline(mod) 
            except Exception as e:
                 print(f"    WARNING: Static shape tuning failed for {model_name}: {e}")
                 print(f"    Proceeding without tuning for {model_name}.")
        else:
            print(f"    Skipping static_shape_tuning for {model_name} (trials=0).")


        print(f"    Building {model_name} runtime module...")
        ex = tvm.compile(mod, target=self.target)
        vm = relax.VirtualMachine(ex, self.device)

        
        
        tvm_params = []
        if "main" in params: 
            for p in params["main"]:
                 if isinstance(p, torch.Tensor):
                      tvm_params.append(tvm.nd.array(p.detach().cpu().numpy(), self.device))
                 elif isinstance(p, tvm.nd.NDArray): 
                      tvm_params.append(p.copyto(self.device))
                 else:
                      print(f"    Warning: Skipping unexpected parameter type in {model_name}: {type(p)}")
        else:
             print(f"    Warning: No 'main' function found in detached params for {model_name}.")


        end_time = time()
        print(f"  {model_name} compilation took {end_time - start_time:.2f} seconds.")
        return vm, tvm_params

    def _profile_tvm_model(self, vm, params, example_inputs_tvm, model_name):
        """Profiles the execution time of a compiled TVM VirtualMachine."""
        if vm is None:
            print(f"Skipping profiling for {model_name} as it was not compiled.")
            return

        print(f"--- Profiling TVM {model_name} ---")
        try:
            
            
            
            report = vm.profile("main", *example_inputs_tvm)
            print(report)
            print(f"--- End {model_name} Profiling ---")
        except Exception as e:
            print(f"  Error during {model_name} profiling: {e}")

    
    def _compile_text_encoder(self):
        
        max_length = self.tokenizer.model_max_length
        
        dummy_input_ids = torch.randint(0, self.tokenizer.vocab_size, (2, max_length), dtype=torch.int64).to(self.torch_device)
        example_inputs = (dummy_input_ids,)
        print("  Attempting Text Encoder compilation (might fail or be slow without specific handling)...")
        try:
            
            
            
            print("  Skipping Text Encoder compilation during refactor (TODO).")
            self.text_encoder_vm, self.text_encoder_params = None, None 
        except Exception as e:
            print(f"  Text Encoder compilation failed: {e}. Skipping.")
            self.text_encoder_vm, self.text_encoder_params = None, None

    def _compile_unet(self):
        
        
        batch_size = 2
        height, width = 512, 512 
        latent_channels = self.unet_pt.config.in_channels
        dummy_sample = torch.randn(
            (batch_size, latent_channels, height // 8, width // 8),
            dtype=torch.float32
        ).to(self.torch_device)
        
        
        
        dummy_timestep = torch.tensor(999, dtype=torch.float32).to(self.torch_device) 
        

        embedding_dim = self.text_encoder_pt.config.hidden_size
        dummy_encoder_hidden_states = torch.randn(batch_size, self.tokenizer.model_max_length, embedding_dim, dtype=torch.float32).to(self.torch_device)

        
        
        
        example_inputs_pt = (dummy_sample, dummy_timestep, dummy_encoder_hidden_states)

        self.unet_vm, self.unet_params = self._compile_model(
            self.unet_pt, example_inputs_pt, "UNet"
        )

        
        if self.unet_vm:
            
            print("Converting example inputs for UNet profiling...")
            example_inputs_tvm = [tvm.nd.array(inp.detach().cpu().numpy(), self.device) for inp in example_inputs_pt]
            self._profile_tvm_model(self.unet_vm, self.unet_params, example_inputs_tvm, "UNet")
        

    def _compile_vae(self):
        
        batch_size = 1
        height, width = 512, 512
        latent_channels = self.unet_pt.config.in_channels 
        dummy_latents = torch.randn(batch_size, latent_channels, height // 8, width // 8, dtype=torch.float32).to(self.torch_device)

        
        example_inputs = (dummy_latents,)
        print("  Attempting VAE Decoder compilation (might fail or be slow)...")
        try:
            
            
            
            print("  Skipping VAE Decoder compilation during refactor (TODO).")
            self.vae_vm, self.vae_params = None, None 
        except Exception as e:
            print(f"  VAE Decoder compilation failed: {e}. Skipping.")
            self.vae_vm, self.vae_params = None, None

    def __call__(self, prompt, negative_prompt="", height=512, width=512, num_inference_steps=25, guidance_scale=7.5, seed=0):
        if self.unet_vm is None or self.unet_params is None: 
             raise RuntimeError("UNet model not compiled. Cannot run inference.")

        
        generator = torch.manual_seed(seed) if seed is not None else None
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        prompts = [prompt] if isinstance(prompt, str) else prompt
        if do_classifier_free_guidance:
            negative_prompts = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
            if len(prompts) != len(negative_prompts):
                raise ValueError("Length of prompts and negative_prompts must match for CFG.")
            prompts_for_encoding = negative_prompts + prompts 
        else:
            prompts_for_encoding = prompts


        
        print("Encoding prompts...")
        start_encode_time = time()
        
        if self.text_encoder_vm is not None:
            print("  (Using TVM Text Encoder - Placeholder Implementation)")
            
            
            
            
            
            
            pass 

        
        with torch.no_grad():
             text_input = self.tokenizer(prompts_for_encoding, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
             text_embeddings_pt = self.text_encoder_pt(text_input.input_ids.to(self.torch_device))[0]

        
        text_embeddings_tvm = tvm.nd.array(text_embeddings_pt.detach().cpu().numpy(), self.device)
        end_encode_time = time()
        print(f"Prompt encoding took {end_encode_time - start_encode_time:.2f} seconds.")


        
        print("Preparing initial latents...")
        latent_channels = self.unet_pt.config.in_channels
        latents_pt = torch.randn(
            (batch_size, latent_channels, height // 8, width // 8),
            generator=generator,
            device=self.torch_device, 
            dtype=torch.float32 
        )
        
        latents_pt = latents_pt * self.scheduler.init_noise_sigma


        
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        print(f"Starting denoising loop for {len(timesteps)} steps...")

        
        total_prep_time = 0
        total_unet_time = 0
        total_guidance_post_time = 0 
        loop_start_time = time()

        for _, t in enumerate(tqdm(timesteps)):
            step_start_time = time()

            
            latent_model_input_pt = torch.cat([latents_pt] * 2) if do_classifier_free_guidance else latents_pt
            
            latent_model_input_pt = self.scheduler.scale_model_input(latent_model_input_pt, t)

            
            
            
            timestep_np = t.detach().cpu().numpy().astype(np.float32)
            
            timestep_tvm = tvm.nd.array(timestep_np, self.device)

            latent_model_input_tvm = tvm.nd.array(latent_model_input_pt.detach().cpu().numpy(), self.device)
            

            prep_end_time = time()
            total_prep_time += (prep_end_time - step_start_time)

            
            unet_start_time = time()
            noise_pred_tvm = self.unet_vm["main"](
                 latent_model_input_tvm,
                 timestep_tvm,
                 text_embeddings_tvm, 
                 *self.unet_params
            )
            
            self.device.sync()
            unet_end_time = time()
            total_unet_time += (unet_end_time - unet_start_time)

            
            guidance_post_start_time = time()
            
            
            noise_pred_pt = torch.from_numpy(noise_pred_tvm[0].numpy()).to(self.torch_device) 

            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_pt.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_pt

            
            latents_pt = self.scheduler.step(noise_pred, t, latents_pt).prev_sample
            guidance_post_end_time = time()
            total_guidance_post_time += (guidance_post_end_time - guidance_post_start_time)

        loop_end_time = time()
        print(f"Denoising loop took {loop_end_time - loop_start_time:.2f} seconds.")
        if len(timesteps) > 0: 
            print("--- Profiling Breakdown (Loop Avg Per Step) ---")
            avg_prep = (total_prep_time / len(timesteps)) * 1000
            avg_unet = (total_unet_time / len(timesteps)) * 1000
            avg_guidance_post = (total_guidance_post_time / len(timesteps)) * 1000
            avg_total = avg_prep + avg_unet + avg_guidance_post
            print(f"  Avg Data Prep Time:    {avg_prep:.2f} ms")
            print(f"  Avg UNet TVM Time:     {avg_unet:.2f} ms")
            print(f"  Avg Guidance+Sched Time: {avg_guidance_post:.2f} ms")
            print(f"  Avg Total Measured:    {avg_total:.2f} ms")


        
        print("Decoding latents...")
        start_decode_time = time()
        
        if self.vae_vm is not None:
             print("  (Using TVM VAE Decoder - Placeholder Implementation)")
             
             
             
             
             
             
             pass 

        
        
        latents_pt = 1 / self.vae_pt.config.scaling_factor * latents_pt
        with torch.no_grad():
            image_pt = self.vae_pt.decode(latents_pt).sample
        image_np = image_pt.detach().cpu().numpy() 

        end_decode_time = time()
        print(f"VAE decoding took {end_decode_time - start_decode_time:.2f} seconds.")


        
        
        image_np = (image_np / 2 + 0.5).clip(0, 1)
        image_np = image_np.transpose((0, 2, 3, 1)) 
        images = [Image.fromarray((img * 255).round().astype("uint8")) for img in image_np]

        print("Inference complete.")
        return images



if __name__ == "__main__":
    print("Initializing TVM Stable Diffusion Pipeline...")
    
    pipeline = TVMStableDiffusionPipeline(tuning_trials=5, device="cuda") 

    prompt = "a photograph of an astronaut riding a horse"
    
    print(f"Generating image for prompt: '{prompt}'")
    start_call_time = time()
    images = pipeline(prompt, num_inference_steps=25, seed=42) 
    end_call_time = time()
    total_time = end_call_time - start_call_time
    print(f"Pipeline execution took {total_time:.2f} seconds.")

    
    if images:
        if len(images) == 1:
             images[0].save("tvm_astronaut_horse.png")
             print("Image saved to tvm_astronaut_horse.png")
        else:
             for i, img in enumerate(images):
                  img_path = f"tvm_generated_image_{i}.png"
                  img.save(img_path)
                  print(f"Image saved to {img_path}")


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
    tvm_device: str = "cuda",
    tuning_trials: int = 0,
    target_str: str = "cuda",
) -> Optional[float]:
    """Runs Stable Diffusion inference using TVM‑compiled models.

    Returns the time taken for the *pipeline call* (does NOT include initial
    compilation time, which can be very large). If compilation or inference
    fails, returns ``None``.
    """

    if TVMStableDiffusionPipeline is None:
        logger.error("TVMStableDiffusionPipeline import failed – skipping TVM benchmark.")
        return None

    
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

    
    if images:
        logger.info("Saving generated TVM image(s)…")
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                try:
                    
                    from PIL import Image as _PILImage

                    img = _PILImage.fromarray(img)
                except Exception:
                    logger.warning("Could not convert image %d to PIL — skipping save.", i)
                    continue
            img.save(f"{output_image_prefix}_tvm_{i}.png")

    
    del pipeline, images  
    gc.collect()

    return infer_time
