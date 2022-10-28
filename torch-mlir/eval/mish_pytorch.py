import torch
from torch import nn


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log(1 + torch.exp(x))
        return x


model = Mish()
model.eval()
data = torch.rand(8, 32, 224, 224)

golden_prediction = model.forward(data)
# print("PyTorch prediction")
# print(golden_prediction)
