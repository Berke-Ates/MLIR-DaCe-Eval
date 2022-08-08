import sys

# sys.path.insert(0, "/home/xdb/dace")

import dace
import json
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace import SDFG

print("Using DaCe from: %s" % dace.__file__)

sdfg = SDFG.from_json(json.load(sys.stdin))
sdfg.validate()

auto_optimize(sdfg, dace.DeviceType.CPU)
ConstantPropagation().apply_pass(sdfg, {})
Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
sdfg.simplify()

sdfg.save(sys.argv[1])
