#ALL: Makefile.CUDA SumReductionRevealed
ALL: Makefile.CUDA OpenCL_Nvidia DPCPP Kokkos Raja

.PHONY: Makefile.CUDA SumReductionRevealed OpenCL_Nvidia DPCPP Kokkos Raja

Makefile.CUDA:
	make -f ./Makefile.CUDA

SumReductionRevealed:
	cd CUDA/SumReductionRevealed && make && ./SumReductionRevealed

Kokkos:
	cd Kokkos/StreamTriad && mkdir build && cd build && cmake .. && make && ./StreamTriad

Raja:
	cd Raja/StreamTriad && mkdir build && cd build && cmake .. && make && ./StreamTriad

OpenCL_Nvidia:
	cd OpenCL/StreamTriad && cmake . && make && ./StreamTriad

DPCPP:
	cd DPCPP/StreamTriad && make && ./StreamTriad

clean:
	#cd CUDA/SumReductionRevealed && make clean
	make -f ./Makefile.CUDA clean
	cd Kokkos/StreamTriad && rm -rf build
	cd Raja/StreamTriad && rm -rf build
	cd OpenCL/StreamTriad && rm -rf build
	cd DPCPP/StreamTriad && make clean
