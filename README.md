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
- `lulesh_driver.cpp` is a modified LULESH benchmark, which uses external calc 
  function provided the `-DUSE_EXTERNAL_CALCS` flag
- `calcs.cpp` contains the extracted calc functions in their unaltered state 
   (just for reference)
- `mod_calcs.cpp` is a modified version of the calc functions
- `script.sh` generates mlir code from `calcs.cpp`
