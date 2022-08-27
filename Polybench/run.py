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


# def print1DArray(arr):
#     for elemIdx, elem in enumerate(arr):
#         if elemIdx % 20 == 0:
#             print("", file=sys.stderr)
#         printDouble(elem)

# def print2DArray(arr):
#     for rowIdx, row in enumerate(arr):
#         for elemIdx, elem in enumerate(row):
#             if (rowIdx * len(arr) + elemIdx) % 20 == 0:
#                 print("", file=sys.stderr)
#             printDouble(elem)

# def print3DArray(arr):
#     for rowIdx, row in enumerate(arr):
#         for colIdx, col in enumerate(row):
#             for elemIdx, elem in enumerate(col):
#                 if (rowIdx * len(row) * len(arr) + colIdx * len(row) +
#                         elemIdx) % 20 == 0:
#                     print("", file=sys.stderr)
#                 printDouble(elem)

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

argDict = {}

for argName, argType in sdfg.arglist().items():
    arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
    argDict[argName] = arr

start_time = time.time()
obj(**argDict)

print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)

for argName, arr in argDict.items():
    print("begin dump: %s" % argName, end='', file=sys.stderr)
    printArray(arr, 0, len(arr.shape))
    print("\nend   dump: %s" % argName, file=sys.stderr)

print("==END   DUMP_ARRAYS==", file=sys.stderr)

print("%d" % int((time.time() - start_time) * 1000))
