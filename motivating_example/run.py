import sys
from dace import SDFG
import time

sdfg = SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

start_time = time.time()
obj()
print("%.2f" % (time.time() - start_time))
