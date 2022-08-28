import sys

sys.path.insert(0, "/home/xdb/dace")

import time
import dace

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

argDict = {}

for argName, argType in sdfg.arglist().items():
    arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
    argDict[argName] = arr

start_time = time.time()
obj()
elapsed = int((time.time() - start_time) * 1000)
print("%d" % elapsed)

# for argName, arr in argDict.items():
#     print(arr)