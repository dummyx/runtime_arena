import argparse
import logging
import sys

from export_models import export_models
from benchmark import run_benchmarks


def main():
    """Main entry point for the runtime arena project.
    Provides a unified interface for exporting models and running benchmarks.
    """
    parser = argparse.ArgumentParser(
        description="OpenXLA Runtime Arena: Export and benchmark Stable Diffusion models with various runtimes"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export models to ONNX and compile to IREE VMFB")
    export_parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", 
                              help="Hugging Face model ID.")
    export_parser.add_argument("--output_dir", type=str, default="./models", 
                              help="Directory to save ONNX and VMFB models.")
    export_parser.add_argument("--batch_size", type=int, default=1, 
                              help="Batch size for dummy inputs and dynamic axes.")
    export_parser.add_argument("--height", type=int, default=512, 
                              help="Image height for dummy inputs.")
    export_parser.add_argument("--width", type=int, default=512, 
                              help="Image width for dummy inputs.")
    export_parser.add_argument("--opset", type=int, default=17, 
                              help="ONNX opset version.")
    export_parser.add_argument("--skip_existing", action='store_true', 
                              help="Skip export/compilation if target files exist.")
    export_parser.add_argument("--no_iree", action='store_true', 
                              help="Do not compile ONNX models to IREE VMFB.")
    export_parser.add_argument("--iree_target", type=str, default="cuda", 
                              help="IREE target backend (e.g., cuda, vulkan, cuda).")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model inference")
    benchmark_parser.add_argument("--model_id", type=str, 
                                default="stable-diffusion-v1-5/stable-diffusion-v1-5", 
                                help="Hugging Face model ID for tokenizer/scheduler.")
    benchmark_parser.add_argument("--model_dir", type=str, default="./models", 
                                help="Directory containing ONNX/VMFB models.")
    benchmark_parser.add_argument("--prompt", type=str, 
                                default="a photo of an astronaut riding a horse on the moon", 
                                help="Inference prompt.")
    benchmark_parser.add_argument("--batch_size", type=int, default=1, 
                                help="Batch size for inference.")
    benchmark_parser.add_argument("--height", type=int, default=512, 
                                help="Image height.")
    benchmark_parser.add_argument("--width", type=int, default=512, 
                                help="Image width.")
    benchmark_parser.add_argument("--steps", type=int, default=50, 
                                help="Number of inference steps.")
    benchmark_parser.add_argument("--guidance", type=float, default=7.5, 
                                help="Guidance scale.")
    benchmark_parser.add_argument("--iree_target", type=str, default="cuda", 
                                help="IREE target backend identifier (for VMFB filename).")
    benchmark_parser.add_argument("--iree_device", type=str, default="local-task", 
                                help="IREE device identifier (e.g., local-task, local-cpu, local-gpu).")
    benchmark_parser.add_argument("--skip_pytorch", action='store_true', 
                                help="Skip PyTorch benchmark.")
    benchmark_parser.add_argument("--skip_onnx", action='store_true', 
                                help="Skip Optimum ONNX benchmark.")
    benchmark_parser.add_argument("--skip_iree", action='store_true', 
                                help="Skip IREE benchmark.")
    benchmark_parser.add_argument("--skip_direct_onnx", action='store_true', 
                                help="Skip Direct ONNX Runtime benchmark.")
    benchmark_parser.add_argument("--skip_tvm", action='store_true', 
                                help="Skip TVM benchmark.")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Dispatch to appropriate command
    if args.command == "export":
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
    elif args.command == "benchmark":
        # We'll implement a run_benchmarks function in benchmark.py
        run_benchmarks(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
