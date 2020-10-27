#ALL: Makefile.CUDA SumReductionRevealed
ALL: Makefile.CUDA Kokkos Raja OpenCL_Nvidia

.PHONY: Makefile.CUDA SumReductionRevealed Kokkos Raja OpenCL_Nvidia

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

clean:
	#cd CUDA/SumReductionRevealed && make clean
	make -f ./Makefile.CUDA clean
	cd Kokkos/StreamTriad && rm -rf build
	cd Raja/StreamTriad && rm -rf build
	cd OpenCL/StreamTriad && make clean
