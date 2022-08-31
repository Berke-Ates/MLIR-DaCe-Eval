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


def printDoitgen(arr):
    NR = 10
    NQ = 8
    NP = 12
    for i in range(NR):
        for j in range(NQ):
            for k in range(NP):
                if (i * NQ * NP + j * NP + k) % 20 == 0:
                    print("", file=sys.stderr)
                print("%.4f " % arr[i, j, k], end='', file=sys.stderr)


def printCholesky(arr):
    N = 40
    for i in range(N):
        for j in range(i + 1):
            if (i * N + j) % 20 == 0:
                print("", file=sys.stderr)
            print("%.4f " % arr[i, j], end='', file=sys.stderr)


def printGramschmidt(arr, useN):
    M = 20
    N = 30
    iUB = N if useN else M

    for i in range(iUB):
        for j in range(N):
            if (i * N + j) % 20 == 0:
                print("", file=sys.stderr)
            print("%.4f " % arr[i, j], end='', file=sys.stderr)


sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

argDict = {}

for argName, argType in sdfg.arglist().items():
    print(argName)
    # if argName == "argv_loc":
    #     arr = dace.ndarray(shape=(43, ), dtype=argType.dtype)
    #     argDict[argName] = arr
    # else:
    #     arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
    #     argDict[argName] = arr

_argv_loc = dace.ndarray(shape=(43, ), dtype=argType.dtype)
argDict["argv"] = _argv_loc

argc = dace.ndarray(shape=(43, ), dtype=argType.dtype)
argDict["argc"] = argc

start_time = time.time()
obj(_argv_loc=_argv_loc, _argcount=43, argc=argc)

print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)

for argName, arr in argDict.items():
    print("begin dump: %s" % argName, end='', file=sys.stderr)
    if "cholesky" in sys.argv[1]:
        printCholesky(arr)
    elif "gramschmidt" in sys.argv[1]:
        printGramschmidt(arr, argName == "_arg1")
    else:
        # printArray(arr, 0, len(arr.shape))
        printDoitgen(arr)

    print("\nend   dump: %s" % argName, file=sys.stderr)

print("==END   DUMP_ARRAYS==", file=sys.stderr)

print("%d" % int((time.time() - start_time) * 1000))
