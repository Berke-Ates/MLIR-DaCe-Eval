import sys

sys.path.insert(0, "/home/xdb/dace")

import dace
import json
from dace.transformation.auto.auto_optimize import auto_optimize, move_small_arrays_to_stack
from dace.transformation.passes.scalar_to_symbol import promote_scalars_to_symbols
from dace.transformation.dataflow.tasklet_fusion import TaskletFusion
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace import SDFG

print("   Using DaCe from: %s" % dace.__file__)

sdfg = SDFG.from_json(json.load(sys.stdin))
sdfg.validate()

# promote_scalars_to_symbols(sdfg, integers_only=False)
sdfg.simplify()
move_small_arrays_to_stack(sdfg)
auto_optimize(sdfg, dace.DeviceType.CPU)
# sdfg.apply_transformations_repeated([TrivialTaskletElimination])
# auto_optimize(sdfg, dace.DeviceType.CPU)

for node, parent in sdfg.all_nodes_recursive():
    if isinstance(node, dace.nodes.MapEntry):
        node.schedule = dace.ScheduleType.Sequential

obj = sdfg.compile()
