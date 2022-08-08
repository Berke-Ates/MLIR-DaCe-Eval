import sys
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
print("%.2f" % (time.time() - start_time))
#print("res: %s" % A)
