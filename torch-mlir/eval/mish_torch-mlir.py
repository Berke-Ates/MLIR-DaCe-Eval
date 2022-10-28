import torch
import sys
from pathlib import Path

from torch_mlir_e2e_test.mhlo_backends.linalg_on_tensors import LinalgOnTensorsMhloBackend

compiled = Path('mish_llvm.mlir').read_text()

backend = LinalgOnTensorsMhloBackend()
jit_module = backend.load(compiled)
jit_func = jit_module.forward

data = torch.rand(8, 32, 224, 224)
prediction = torch.from_numpy(jit_func(data.numpy()))
print("torch-mlir prediction")
print(prediction)
