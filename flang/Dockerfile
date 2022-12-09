FROM ubuntu:latest as system
WORKDIR /home

# Install minimal tools + default .bashrc
RUN apt update -y && apt upgrade -y && apt install -y git command-not-found

RUN git clone https://gist.github.com/1637696.git \
  && mv 1637696/.bashrc . && rm -rf 1637696

ENV TERM=xterm-color

ONBUILD RUN apt -qq update

# Run additional setup
FROM system as flang

RUN apt install -y cmake ninja-build clang lld

RUN git clone --recurse-submodules --depth 1 --shallow-submodules \
  https://github.com/llvm/llvm-project
WORKDIR /home/llvm-project

RUN mkdir build
WORKDIR /home/llvm-project/build
RUN cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  # -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$LD_LIBRARY_PATH" \
  -DFLANG_ENABLE_WERROR=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_LIT_ARGS=-v \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;flang;openmp" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt"

RUN ninja check-flang -j1
RUN ninja install -j1
RUN ninja -j1

WORKDIR /home
ENV PATH="$PATH:/home/llvm-project/build/bin/"
CMD bash
