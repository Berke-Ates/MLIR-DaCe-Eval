import sys

sys.path.insert(0, "/home/xdb/dace")

import time
import dace


def printArray(arr, offset, depth):
    if (depth > 0):
        for dimIdx, dim in enumerate(arr):
            offsetFac = len(arr) if depth > 1 else 1
            printArray(dim, offsetFac * (offset + dimIdx), depth - 1)
    else:
        if offset % 20 == 0:
            print("", file=sys.stderr)
        print("%.4f " % arr, end='', file=sys.stderr)


sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

for i in range(10):
    argDict = {}

    for argName, argType in sdfg.arglist().items():
        arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
        argDict[argName] = arr

    obj(**argDict)
