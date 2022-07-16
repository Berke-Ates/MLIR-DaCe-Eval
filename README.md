# MLIR-SDFG Evaluation on LULESH
This is a repository to evaluate MLIR-SDFG on the LULESH benchmark.

## Prerequisites
You need the following tools installed in order to run this evaluation:
- mlir-sdfg
- Polygeist
- clang & clang++
- LLD

Build MLIR with:
```shell
cmake -G Ninja ../llvm \
 -DLLVM_ENABLE_PROJECTS="mlir;clang " \
 -DLLVM_TARGETS_TO_BUILD="host" \
 -DLLVM_ENABLE_ASSERTIONS=ON \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER=clang \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DLLVM_ENABLE_LLD=ON \
```

## Files
- `normal_output.txt` contains the output of a regular run
- `lulesh_driver.cpp` is a modified LULESH benchmark, which uses external 
  functions provided the `-DUSE_EXTERNAL_<function>` flag
- `calcs.cpp` contains the extracted calc functions in their unaltered state 
   (just for reference)
- `ext/` contains all the functions in their altered form
- `run.sh ext/<function>.cpp` compiles the provided function using Polygeist,
links with the driver (settings the `-DUSE_EXTERNAL_<function>` flag), runs the
result and checks with `normal_output.txt` for correctness

