FROM ubuntu:22.10
WORKDIR /home

COPY ./clang/15.0.0 /home/lib/clang/15.0.0
COPY ./bin ./bin
# COPY llvm_bitcodes-10.0.0-poj104-v1.tar.gz .
COPY ./ProgramData ./ProgramData
# COPY add_includes.sh .
COPY check.sh .

RUN apt update -y
# RUN apt install -y gcc g++ python-is-python3
RUN apt install -y bc
# RUN tar -xvzf llvm_bitcodes-10.0.0-poj104-v1.tar.gz
RUN chmod +x check.sh
# RUN chmod +x add_includes.sh
# RUN ./add_includes.sh

ENV PATH="/home/bin:$PATH"

CMD ["bash"]

