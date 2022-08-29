import sys

sys.path.insert(0, "/home/xdb/dace")

import time
import dace

# To check the output incomment all the comments and change the data container
# in the sdfg from scalar to array

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

for i in range(10):
    argDict = {}

    for argName, argType in sdfg.arglist().items():
        arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
        argDict[argName] = arr

    obj(**argDict)
