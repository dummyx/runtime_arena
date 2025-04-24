# Add necessary imports if they are missing at the top
import torch
from torch.export import export
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.frontend import detach_params
# from tvm.contrib import tvmjs # May be needed for compilation artifacts later
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from time import time

class TVMStableDiffusionPipeline:
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", device="cpu", target_str="llvm -num-cores 16 -mcpu=zen4", tuning_trials=80):
        self.device = tvm.device("cpu", 0) if device == "cpu" else tvm.cuda(0) # Basic device handling
        self.target = tvm.target.Target(target_str)
        self.tuning_trials = tuning_trials
        self.torch_device = torch.device(device)

        print("Loading models...")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        # Load PyTorch models first
        self.text_encoder_pt = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", use_safetensors=True).to(self.torch_device)
        self.unet_pt = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_safetensors=True).to(self.torch_device)
        self.vae_pt = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True).to(self.torch_device)

        # Initialize placeholders for compiled models
        self.text_encoder_vm, self.text_encoder_params = None, None
        self.unet_vm, self.unet_params = None, None
        self.vae_vm, self.vae_params = None, None

        # We will call compilation methods later
        print("Compiling models with TVM...")
        self._compile_text_encoder()
        self._compile_unet()
        self._compile_vae()
        print("TVM compilation complete.")

    def _compile_model(self, pt_model, example_inputs, model_name):
        print(f"  Compiling {model_name}...")
        start_time = time()

        print(f"    Exporting {model_name}...")
        # Ensure model is in eval mode for export
        pt_model.eval()
        exported_program = export(pt_model, example_inputs)

        print(f"    Converting {model_name} to Relax...")
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        mod, params = detach_params(mod)

        print(f"    Applying static_shape_tuning to {model_name}...")
        # Check if tuning trials > 0
        if self.tuning_trials > 0:
            pipeline = relax.get_pipeline("static_shape_tuning", target=self.target, total_trials=self.tuning_trials)
            try:
                mod = pipeline(mod) # Apply tuning pipeline
            except Exception as e:
                 print(f"    WARNING: Static shape tuning failed for {model_name}: {e}")
                 print(f"    Proceeding without tuning for {model_name}.")
        else:
            print(f"    Skipping static_shape_tuning for {model_name} (trials=0).")


        print(f"    Building {model_name} runtime module...")
        ex = tvm.compile(mod, target=self.target)
        vm = relax.VirtualMachine(ex, self.device)

        # Convert params to TVM NDArrays
        # Make sure parameters are detached and moved to CPU before converting to numpy
        tvm_params = []
        if "main" in params: # Check if 'main' function exists in detached params
            for p in params["main"]:
                 if isinstance(p, torch.Tensor):
                      tvm_params.append(tvm.nd.array(p.detach().cpu().numpy(), self.device))
                 elif isinstance(p, tvm.nd.NDArray): # If already an NDArray (less likely here)
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
            # vm.profile expects inputs first, then parameters.
            # The example_inputs_tvm should be a list/tuple of TVM NDArrays.
            # Use a reasonable number of repeats for profiling, e.g., 10
            report = vm.profile("main", *example_inputs_tvm)
            print(report)
            print(f"--- End {model_name} Profiling ---")
        except Exception as e:
            print(f"  Error during {model_name} profiling: {e}")

    # --- Placeholder for other methods ---
    def _compile_text_encoder(self):
        # Placeholder - requires correct example input shapes
        max_length = self.tokenizer.model_max_length
        # Use batch size 2 for CFG consistency if text encoder is used in loop
        dummy_input_ids = torch.randint(0, self.tokenizer.vocab_size, (2, max_length), dtype=torch.int64).to(self.torch_device)
        example_inputs = (dummy_input_ids,)
        print("  Attempting Text Encoder compilation (might fail or be slow without specific handling)...")
        try:
            # self.text_encoder_vm, self.text_encoder_params = self._compile_model(
            #     self.text_encoder_pt, example_inputs, "Text Encoder"
            # )
            print("  Skipping Text Encoder compilation during refactor (TODO).")
            self.text_encoder_vm, self.text_encoder_params = None, None # Keep as placeholder
        except Exception as e:
            print(f"  Text Encoder compilation failed: {e}. Skipping.")
            self.text_encoder_vm, self.text_encoder_params = None, None

    def _compile_unet(self):
        # Example inputs based on typical Stable Diffusion usage
        # Use batch size 2 for CFG
        batch_size = 2
        height, width = 512, 512 # Assuming standard 512x512, adjust if needed
        latent_channels = self.unet_pt.config.in_channels
        dummy_sample = torch.randn(
            (batch_size, latent_channels, height // 8, width // 8),
            dtype=torch.float32
        ).to(self.torch_device)
        # Timestep needs correct type/shape - check model's forward pass signature
        # diffusers UNet typically expects a scalar or 1D tensor timestep. torch.export might require a tensor.
        # Let's try with a 0-D tensor first.
        dummy_timestep = torch.tensor(999, dtype=torch.float32).to(self.torch_device) # Using float32 as often expected by compiled models
        # dummy_timestep = torch.tensor([999]*batch_size, dtype=torch.int64).to(self.torch_device) # Alternative if int64 or 1D tensor needed

        embedding_dim = self.text_encoder_pt.config.hidden_size
        dummy_encoder_hidden_states = torch.randn(batch_size, self.tokenizer.model_max_length, embedding_dim, dtype=torch.float32).to(self.torch_device)

        # Ensure example inputs match the exact signature expected by unet_pt.forward
        # Check UNet source or docs if errors occur during export.
        # Common signature: sample, timestep, encoder_hidden_states, optional args...
        example_inputs_pt = (dummy_sample, dummy_timestep, dummy_encoder_hidden_states)

        self.unet_vm, self.unet_params = self._compile_model(
            self.unet_pt, example_inputs_pt, "UNet"
        )

        # --- Add Profiling Call ---
        if self.unet_vm:
            # Convert example inputs to TVM NDArrays for profiling
            print("Converting example inputs for UNet profiling...")
            example_inputs_tvm = [tvm.nd.array(inp.detach().cpu().numpy(), self.device) for inp in example_inputs_pt]
            self._profile_tvm_model(self.unet_vm, self.unet_params, example_inputs_tvm, "UNet")
        # --- End Profiling Call ---

    def _compile_vae(self):
        # Placeholder - VAE decoding might have specific needs for export
        batch_size = 1
        height, width = 512, 512
        latent_channels = self.unet_pt.config.in_channels # Use UNet's channel count
        dummy_latents = torch.randn(batch_size, latent_channels, height // 8, width // 8, dtype=torch.float32).to(self.torch_device)

        # Wrap the decode call in a simple nn.Module for export
        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
                # Ensure scaling factor is treated as a buffer or constant if needed
                self.register_buffer('scaling_factor', torch.tensor(self.vae.config.scaling_factor))

            def forward(self, x):
                 # Scale factor from original SD pipeline - ensure it works with export
                 x = (1 / self.scaling_factor) * x
                 # Use .sample if needed, or just the direct output
                 decoded = self.vae.decode(x)
                 return decoded.sample # or just decoded depending on what you need post-compilation

        vae_decoder_wrapper = VAEDecoderWrapper(self.vae_pt).to(self.torch_device).eval()
        example_inputs = (dummy_latents,)
        print("  Attempting VAE Decoder compilation (might fail or be slow)...")
        try:
            # self.vae_vm, self.vae_params = self._compile_model(
            #    vae_decoder_wrapper, example_inputs, "VAE Decoder"
            # )
            print("  Skipping VAE Decoder compilation during refactor (TODO).")
            self.vae_vm, self.vae_params = None, None # Keep as placeholder
        except Exception as e:
            print(f"  VAE Decoder compilation failed: {e}. Skipping.")
            self.vae_vm, self.vae_params = None, None

    def __call__(self, prompt, negative_prompt="", height=512, width=512, num_inference_steps=25, guidance_scale=7.5, seed=0):
        if self.unet_vm is None or self.unet_params is None: # Check UNet is compiled
             raise RuntimeError("UNet model not compiled. Cannot run inference.")

        # --- 1. Input Processing ---
        generator = torch.manual_seed(seed) if seed is not None else None
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        prompts = [prompt] if isinstance(prompt, str) else prompt
        if do_classifier_free_guidance:
            negative_prompts = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
            if len(prompts) != len(negative_prompts):
                raise ValueError("Length of prompts and negative_prompts must match for CFG.")
            prompts_for_encoding = negative_prompts + prompts # Order matters for CFG splitting later
        else:
            prompts_for_encoding = prompts


        # --- 2. Prompt Encoding (Using PyTorch for now) ---
        print("Encoding prompts...")
        start_encode_time = time()
        # TODO: Replace with TVM Text Encoder when compiled and validated
        if self.text_encoder_vm is not None:
            print("  (Using TVM Text Encoder - Placeholder Implementation)")
            # Example:
            # text_input = self.tokenizer(prompts_for_encoding, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="np") # Get numpy arrays directly if possible
            # text_input_ids_tvm = tvm.nd.array(text_input.input_ids, self.device)
            # text_embeddings_tvm = self.text_encoder_vm["main"](text_input_ids_tvm, *self.text_encoder_params)
            # # Need to handle splitting for CFG if TVM encoder outputs combined embeddings
            # text_embeddings_pt = torch.from_numpy(text_embeddings_tvm.numpy()).to(self.torch_device) # Convert back for now if needed later
            pass # Pass until TVM encoder is ready

        # Using PyTorch Text Encoder as fallback/default
        with torch.no_grad():
             text_input = self.tokenizer(prompts_for_encoding, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
             text_embeddings_pt = self.text_encoder_pt(text_input.input_ids.to(self.torch_device))[0]

        # Convert embeddings to TVM format *once* before the loop
        text_embeddings_tvm = tvm.nd.array(text_embeddings_pt.detach().cpu().numpy(), self.device)
        end_encode_time = time()
        print(f"Prompt encoding took {end_encode_time - start_encode_time:.2f} seconds.")


        # --- 3. Prepare Initial Latents ---
        print("Preparing initial latents...")
        latent_channels = self.unet_pt.config.in_channels
        latents_pt = torch.randn(
            (batch_size, latent_channels, height // 8, width // 8),
            generator=generator,
            device=self.torch_device, # Keep latents on torch device for scheduler
            dtype=torch.float32 # Ensure correct dtype
        )
        # Scale the initial noise by the standard deviation required by the scheduler
        latents_pt = latents_pt * self.scheduler.init_noise_sigma


        # --- 4. Denoising Loop ---
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        print(f"Starting denoising loop for {len(timesteps)} steps...")

        # Initialize profilers
        total_prep_time = 0
        total_unet_time = 0
        total_guidance_post_time = 0 # Guidance calculation + scheduler step
        loop_start_time = time()

        for _, t in enumerate(tqdm(timesteps)):
            step_start_time = time()

            # Expand latents for CFG (if needed)
            latent_model_input_pt = torch.cat([latents_pt] * 2) if do_classifier_free_guidance else latents_pt
            # Scale model input according to scheduler
            latent_model_input_pt = self.scheduler.scale_model_input(latent_model_input_pt, t)

            # Prepare inputs for TVM UNet
            # Convert timestep - ensure it's the type/shape the compiled model expects
            # Try converting to float32 numpy scalar/0-d array first
            timestep_np = t.detach().cpu().numpy().astype(np.float32)
            # timestep_np = np.array(timestep_np) # Ensure it's a 0-d array if needed
            timestep_tvm = tvm.nd.array(timestep_np, self.device)

            latent_model_input_tvm = tvm.nd.array(latent_model_input_pt.detach().cpu().numpy(), self.device)
            # We use the text_embeddings_tvm prepared before the loop

            prep_end_time = time()
            total_prep_time += (prep_end_time - step_start_time)

            # --- Run UNet inference ---
            unet_start_time = time()
            noise_pred_tvm = self.unet_vm["main"](
                 latent_model_input_tvm,
                 timestep_tvm,
                 text_embeddings_tvm, # Pass the combined embeddings
                 *self.unet_params
            )
            # Crucial: Ensure computation is finished before stopping timer, especially on GPU
            self.device.sync()
            unet_end_time = time()
            total_unet_time += (unet_end_time - unet_start_time)

            # --- Process Output & Scheduler Step ---
            guidance_post_start_time = time()
            # Convert result back to PyTorch for guidance and scheduler
            # Access the first element of the returned container
            noise_pred_pt = torch.from_numpy(noise_pred_tvm[0].numpy()).to(self.torch_device) # Move back to torch device

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_pt.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_pt

            # Scheduler step (operates on PyTorch tensors)
            latents_pt = self.scheduler.step(noise_pred, t, latents_pt).prev_sample
            guidance_post_end_time = time()
            total_guidance_post_time += (guidance_post_end_time - guidance_post_start_time)

        loop_end_time = time()
        print(f"Denoising loop took {loop_end_time - loop_start_time:.2f} seconds.")
        if len(timesteps) > 0: # Avoid division by zero
            print("--- Profiling Breakdown (Loop Avg Per Step) ---")
            avg_prep = (total_prep_time / len(timesteps)) * 1000
            avg_unet = (total_unet_time / len(timesteps)) * 1000
            avg_guidance_post = (total_guidance_post_time / len(timesteps)) * 1000
            avg_total = avg_prep + avg_unet + avg_guidance_post
            print(f"  Avg Data Prep Time:    {avg_prep:.2f} ms")
            print(f"  Avg UNet TVM Time:     {avg_unet:.2f} ms")
            print(f"  Avg Guidance+Sched Time: {avg_guidance_post:.2f} ms")
            print(f"  Avg Total Measured:    {avg_total:.2f} ms")


        # --- 5. VAE Decoding (Using PyTorch for now) ---
        print("Decoding latents...")
        start_decode_time = time()
        # TODO: Replace with TVM VAE Decoder when compiled and validated
        if self.vae_vm is not None:
             print("  (Using TVM VAE Decoder - Placeholder Implementation)")
             # latents_tvm = tvm.nd.array(latents_pt.cpu().numpy(), self.device)
             # # Handle scaling if not done in the wrapper during compilation
             # scaling_factor = self.vae_pt.config.scaling_factor
             # latents_tvm = relax.op.multiply(latents_tvm, relax.const(1.0 / scaling_factor, "float32")) # This requires relax context
             # image_tvm = self.vae_vm["main"](latents_tvm, *self.vae_params)
             # image_np = image_tvm.numpy()
             pass # Pass until TVM VAE is ready

        # Using PyTorch VAE Decoder as fallback/default
        # Scale latents before decoding
        latents_pt = 1 / self.vae_pt.config.scaling_factor * latents_pt
        with torch.no_grad():
            image_pt = self.vae_pt.decode(latents_pt).sample
        image_np = image_pt.detach().cpu().numpy() # Keep as numpy array

        end_decode_time = time()
        print(f"VAE decoding took {end_decode_time - start_decode_time:.2f} seconds.")


        # --- 6. Post-processing ---
        # Convert to PIL Images
        image_np = (image_np / 2 + 0.5).clip(0, 1)
        image_np = image_np.transpose((0, 2, 3, 1)) # Channel-last format for PIL
        images = [Image.fromarray((img * 255).round().astype("uint8")) for img in image_np]

        print("Inference complete.")
        return images


# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing TVM Stable Diffusion Pipeline...")
    # Use fewer tuning trials for faster initialization during testing
    pipeline = TVMStableDiffusionPipeline(tuning_trials=5, device="cpu") # Specify device if not cpu

    prompt = "a photograph of an astronaut riding a horse"
    # prompt = ["a photograph of an astronaut riding a horse", "a painting of a cat sitting on a chair"] # Example batch
    print(f"Generating image for prompt: '{prompt}'")
    start_call_time = time()
    images = pipeline(prompt, num_inference_steps=25, seed=42) # Use a fixed seed
    end_call_time = time()
    total_time = end_call_time - start_call_time
    print(f"Pipeline execution took {total_time:.2f} seconds.")

    # Save the image(s)
    if images:
        if len(images) == 1:
             images[0].save("tvm_astronaut_horse.png")
             print("Image saved to tvm_astronaut_horse.png")
        else:
             for i, img in enumerate(images):
                  img_path = f"tvm_generated_image_{i}.png"
                  img.save(img_path)
                  print(f"Image saved to {img_path}")