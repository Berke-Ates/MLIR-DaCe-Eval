from ast import arg
import sys

sys.path.insert(0, "/home/xdb/dace")

import time
import dace
import os
from dace.sdfg.utils import load_precompiled_sdfg

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

for i in range(10):

    argDict = {}

    for argName, argType in sdfg.arglist().items():
        arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
        argDict[argName] = arr

    obj(**argDict)

# for argName, arr in argDict.items():
#     print(arr)
