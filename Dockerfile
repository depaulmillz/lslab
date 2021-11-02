FROM depaulmillz/cudadev:latest

COPY . /lslab

RUN git clone https://github.com/depaulmillz/UnifiedMemoryGroupAllocation.git

CMD nvidia-smi && cd /UnifiedMemoryGroupAllocation && conan create . && cd /lslab && mkdir build && cd build && cmake .. && make -j4 && ctest
