FROM ubuntu:latest as system
WORKDIR /home

# Install minimal tools + default .bashrc
RUN apt update -y && apt upgrade -y && apt install -y git command-not-found

RUN git clone https://gist.github.com/1637696.git \
  && mv 1637696/.bashrc . && rm -rf 1637696

ENV TERM=xterm-color

ONBUILD RUN apt -qq update

# Run additional setup
FROM system as torch-mlir

RUN apt install -y cmake ninja-build clang lld python3-pip

RUN git clone --recurse-submodules --depth 1 --shallow-submodules \
  https://github.com/llvm/torch-mlir
WORKDIR /home/torch-mlir

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="$PWD"/externals/llvm-external-projects/torch-mlir-dialects \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DLIBTORCH_SRC_BUILD=ON \
  -DLIBTORCH_VARIANT=shared \
  externals/llvm-project/llvm

WORKDIR /home/torch-mlir/build
RUN ninja

WORKDIR /home
ENV PYTHONPATH=/home/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:/home/torch-mlir/examples
ENV PATH="$PATH:/home/torch-mlir/build/bin/"
CMD bash
