import sys

sys.path.insert(0, "/home/xdb/dace")

import time
import dace

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

for i in range(int(sys.argv[2])):
    argDict = {}

    for argName, argType in sdfg.arglist().items():
        arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
        argDict[argName] = arr

    obj(**argDict)
