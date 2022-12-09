import sys
import numpy as np
import time
import numpy as np
import tqdm

sys.path.insert(0, "/home/xdb/dace")
import dace

sdfg = dace.SDFG.from_file("mish.sdfg")
obj = sdfg.compile()

for i in range(10):
    # warmup
    arr = np.random.rand(8, 32, 224, 224).astype(np.float32)
    arg1 = np.random.rand(8, 32, 224, 224).astype(np.float32)
    obj(___constant_8x32x224x224xf32_1=None, _arg0=arr, _arg1=arg1)

times = np.zeros(100)
for i in tqdm.trange(100):
    arr = np.random.rand(8, 32, 224, 224).astype(np.float32)
    arg1 = np.random.rand(8, 32, 224, 224).astype(np.float32)
    start = time.time()
    obj(___constant_8x32x224x224xf32_1=None, _arg0=arr, _arg1=arg1)
    times[i] = time.time() - start

print('Median time:', np.median(times))
