FROM ubuntu:latest as system
WORKDIR /home

# Install minimal tools + default .bashrc
RUN apt update -y && apt upgrade -y && apt install -y git command-not-found

RUN git clone https://gist.github.com/1637696.git \
  && mv 1637696/.bashrc . && rm -rf 1637696

ENV TERM=xterm-color

ONBUILD RUN apt -qq update

# Run additional setup
FROM system as mlir-hlo

RUN apt install -y cmake ninja-build clang lld

RUN git clone https://github.com/tensorflow/mlir-hlo 
WORKDIR /home/mlir-hlo 

RUN git clone https://github.com/llvm/llvm-project.git
WORKDIR /home/mlir-hlo/llvm-project

RUN git checkout $(cat ../build_tools/llvm_version.txt)
WORKDIR /home/mlir-hlo 
RUN build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build

RUN mkdir build
WORKDIR /home/mlir-hlo/build

RUN cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir

RUN ninja check-mlir-hlo

RUN apt install -y python3-pip
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cpu]"

WORKDIR /home
ENV PATH="/home/mlir-hlo/build/bin:$PATH"
ENV PATH="/home/mlir-hlo/llvm-build/bin:$PATH"

CMD bash
