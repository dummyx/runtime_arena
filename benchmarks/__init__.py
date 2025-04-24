"""Benchmark inference modules package."""

from .iree_inference import run_iree_inference
from .pytorch_inference import run_pytorch_inference
from .optimum_onnx_inference import run_optimum_onnx_inference
from .direct_onnx_inference import run_direct_onnx_inference
from .tvm_inference import run_tvm_inference
