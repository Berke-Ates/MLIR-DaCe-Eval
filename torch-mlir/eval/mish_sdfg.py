import sys
import numpy as np

sys.path.insert(0, "/home/xdb/dace")
import dace

sdfg = dace.SDFG.from_file("mish.sdfg")
obj = sdfg.compile()

for i in range(10):

    argDict = {}

    for argName, argType in sdfg.arglist().items():
        arr = np.random.rand(8, 32, 224, 224).astype(np.float32)
        argDict[argName] = arr

    obj(**argDict)

# for argName, arr in argDict.items():
#     print(arr)
