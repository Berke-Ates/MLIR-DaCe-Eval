#!/bin/bash

# Usage: ./compile_mlir.sh <mlir file>

mlir-opt --affine-loop-invariant-code-motion --lower-affine $1 | \
sdfg-opt --convert-to-sdfg | sdfg-translate --mlir-to-sdfg | \
python3 run.py
