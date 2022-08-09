import sys

# sys.path.insert(0, "/home/xdb/dace")

import time
import dace

n = 180
m = 220
arr = "D"

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()
A = dace.ndarray(shape=(180, 220), dtype=dace.float64)

start_time = time.time()
obj(_arg0=A)
elapsed = int((time.time() - start_time) * 1000)

print("%d" % elapsed)

print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)

print("begin dump: %s" % arr, file=sys.stderr)

for ni in range(n):
    for mi in range(m):
        print("%.2f " % A[ni, mi], end='', file=sys.stderr)
    print("", file=sys.stderr)

print("end   dump: %s" % arr, file=sys.stderr)

print("==END   DUMP_ARRAYS==", file=sys.stderr)
