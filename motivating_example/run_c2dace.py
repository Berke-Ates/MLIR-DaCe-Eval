import sys

sys.path.insert(0, "/home/xdb/dace")

import time
import dace

sdfg = dace.SDFG.from_file(sys.argv[1])
obj = sdfg.compile()

for i in range(int(sys.argv[2])):
    argv_loc = dace.ndarray(shape=(0, ), dtype=dace.dtypes.int8)
    obj(argc_loc=0, _argcount=0, argv_loc=argv_loc)
