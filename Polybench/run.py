import sys

# sys.path.insert(0, "/home/xdb/dace")

import time
import dace


def print1DArray(arr):
    elemCnt = 0
    for elem in arr:
        if elemCnt % 20 == 0:
            print("", file=sys.stderr)
        print("%.2f " % elem, end='', file=sys.stderr)
        elemCnt = elemCnt + 1


def print2DArray(arr):
    elemCnt = 0
    for row in arr:
        for elem in row:
            if elemCnt % 20 == 0:
                print("", file=sys.stderr)
            print("%.2f " % elem, end='', file=sys.stderr)
            elemCnt = elemCnt + 1


def print3DArray(arr):
    elemCnt = 0
    for row in arr:
        for col in row:
            for elem in col:
                if elemCnt % 20 == 0:
                    print("", file=sys.stderr)
                print("%.2f " % elem, end='', file=sys.stderr)
                elemCnt = elemCnt + 1


sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

args = []
argNames = []
argDict = {}

for argName, argType in sdfg.arglist().items():
    arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
    args.append(arr)
    argNames.append(argName)
    argDict[argName] = arr

start_time = time.time()
obj(**argDict)

print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)

for i in range(len(args)):
    print("begin dump: %s" % argNames[i], end='', file=sys.stderr)
    if len(args[i].shape) == 1:
        print1DArray(args[i])

    if len(args[i].shape) == 2:
        print2DArray(args[i])

    if len(args[i].shape) == 3:
        print3DArray(args[i])
    print("\nend   dump: %s" % argNames[i], file=sys.stderr)

print("==END   DUMP_ARRAYS==", file=sys.stderr)

elapsed = int((time.time() - start_time) * 1000)
print("%d" % elapsed)
