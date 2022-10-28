import torch
import torch_mlir
import sys
from pathlib import Path

from torch_mlir_e2e_test.mhlo_backends.linalg_on_tensors import LinalgOnTensorsMhloBackend


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log(1 + torch.exp(x))
        return x


model = Mish()
model.eval()
data = torch.rand(8, 32, 224, 224)
backend = LinalgOnTensorsMhloBackend()

module = torch_mlir.compile(model,
                            data,
                            output_type=torch_mlir.OutputType.MHLO)

compiled = backend.compile(module)

jit_module = backend.load(compiled)
jit_func = jit_module.forward

data = torch.rand(8, 32, 224, 224)
prediction = torch.from_numpy(jit_func(data.numpy()))
print("torch-mlir prediction")
print(prediction)
