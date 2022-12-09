import torch
from torch import nn
import time
import numpy as np
import tqdm


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log(1 + torch.exp(x))
        return x


model = Mish()
model.eval()

for i in range(10):
    # warmup
    data = torch.rand(8, 32, 224, 224)
    model.forward(data)

times = np.zeros(100)
for i in tqdm.trange(100):
    data = torch.rand(8, 32, 224, 224)
    start = time.time()
    prediction = model.forward(data)
    times[i] = time.time() - start

print('Median time:', np.median(times))

# golden_prediction = model.forward(data)
# print("PyTorch prediction")
# print(golden_prediction)
