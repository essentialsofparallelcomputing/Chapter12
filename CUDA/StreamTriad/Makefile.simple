all: StreamTriad

NVCC = nvcc
#NVCC_FLAGS = -arch=sm_30
# Can be either in lib or lib64
CUDA_LIB=`which nvcc | sed -e 's!/bin/nvcc!!'`/lib
CUDA_LIB64=`which nvcc | sed -e 's!/bin/nvcc!!'`/lib64

%.o : %.cu
	${NVCC} ${NVCC_FLAGS} -c $< -o $@

StreamTriad: StreamTriad.o timer.o
	${CXX} -o $@ $^ -L${CUDA_LIB} -L${CUDA_LIB64} -lcudart

clean:
	rm -rf StreamTriad *.o Makefile
