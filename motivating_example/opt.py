import sys

sys.path.insert(0, "/home/xdb/dace")

import dace
import json
from dace.transformation.auto.auto_optimize import auto_optimize, move_small_arrays_to_stack
from dace import SDFG

print("   Using DaCe from: %s" % dace.__file__)

sdfg = SDFG.from_json(json.load(sys.stdin))
sdfg.validate()

sdfg.simplify()
move_small_arrays_to_stack(sdfg)
auto_optimize(sdfg, dace.DeviceType.CPU)

for node, parent in sdfg.all_nodes_recursive():
    if isinstance(node, dace.nodes.MapEntry):
        node.schedule = dace.ScheduleType.Sequential

sdfg.instrument = dace.InstrumentationType.Timer

sdfg.save(sys.argv[1])
