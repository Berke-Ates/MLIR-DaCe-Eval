import sys

# sys.path.insert(0, "/home/xdb/dace")

import time
import dace

# To check the output incomment all the comments and change the data container
# in the sdfg from scalar to array

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()
# A = dace.scalar(dtype=dace.int32)

start_time = time.time()
obj(
    #_arg0=A
)
elapsed = int((time.time() - start_time) * 1000)
print("%d" % elapsed)
#print("res: %s" % A)
