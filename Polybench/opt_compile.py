import sys

sys.path.insert(0, "/home/xdb/dace")

import dace
import json
import time
from dace.transformation.auto.auto_optimize import auto_optimize, move_small_arrays_to_stack
from dace.transformation.passes.scalar_to_symbol import promote_scalars_to_symbols
from dace.transformation.dataflow.tasklet_fusion import TaskletFusion
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace import SDFG

sdfg = SDFG.from_file(sys.argv[1])

opt_start = time.time()
sdfg.simplify()
move_small_arrays_to_stack(sdfg)
auto_optimize(sdfg, dace.DeviceType.CPU)

for node, parent in sdfg.all_nodes_recursive():
    if isinstance(node, dace.nodes.MapEntry):
        node.schedule = dace.ScheduleType.Sequential

opt_time = time.time() - opt_start

f = open(sys.argv[2], "a")
f.write(str(opt_time))
f.close()

compile_start = time.time()
obj = sdfg.compile()
compile_time = time.time() - compile_start

f = open(sys.argv[3], "a")
f.write(str(compile_time))
f.close()
