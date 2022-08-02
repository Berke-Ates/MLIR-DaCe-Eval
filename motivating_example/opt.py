import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace import SDFG


sdfg = SDFG.from_file("out.sdfg")
sdfg.validate()
# auto_optimize(sdfg, dace.DeviceType.CPU)
Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
# sdfg.simplify()
# auto_optimize(sdfg, dace.DeviceType.CPU)
# sdfg.simplify()
# Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
sdfg.save("out_opt.sdfg")
# obj = sdfg.compile()
# obj()

