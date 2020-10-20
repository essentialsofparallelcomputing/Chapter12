#ALL: Makefile.CUDA SumReductionRevealed
ALL: Makefile.CUDA

.PHONY: Makefile.CUDA SumReductionRevealed

Makefile.CUDA:
	make -f ./Makefile.CUDA

SumReductionRevealed:
	cd CUDA/SumReductionRevealed && make && ./SumReductionRevealed

clean:
	#cd CUDA/SumReductionRevealed && make clean
	make -f ./Makefile.CUDA clean
