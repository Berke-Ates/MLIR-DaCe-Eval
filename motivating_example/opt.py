import sys

# sys.path.insert(0, "/home/xdb/dace")

import dace
import json
from dace.transformation.auto.auto_optimize import auto_optimize
from dace import SDFG

print("   Using DaCe from: %s" % dace.__file__)

sdfg = SDFG.from_json(json.load(sys.stdin))
sdfg.validate()

sdfg.simplify()
auto_optimize(sdfg, dace.DeviceType.CPU)

sdfg.save(sys.argv[1])
