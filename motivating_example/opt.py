import dace
import json
import sys
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace import SDFG
import time

sdfg = SDFG.from_json(json.load(sys.stdin))
sdfg.validate()

auto_optimize(sdfg, dace.DeviceType.CPU)
ConstantPropagation().apply_pass(sdfg, {})
Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
sdfg.simplify()

obj = sdfg.compile()
A = dace.scalar(dtype=dace.int32)

start_time = time.time()
obj()

print("%s seconds" % (time.time() - start_time))
print("res: %s" % A)
