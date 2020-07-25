ALL: Makefile.CUDA

.PHONY: Makefile.CUDA

Makefile.CUDA:
	make -f ./Makefile.CUDA

SumReductionRevealed:
	cd CUDA/SumReductionRevealed && mkdir build && cd build && cmake .. && make && ./SumReductionRevealed

clean:
	make -f ./Makefile.CUDA clean
	cd CUDA/SumReductionRevealed && rm -rf build
