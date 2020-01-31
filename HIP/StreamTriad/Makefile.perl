all: StreamTriad

CXX = hipcc

%.cc : %.cu
	hipify-perl $^ > $@

StreamTriad.o: StreamTriad.cc

StreamTriad: StreamTriad.o timer.o
	${CXX} -o $@ $^ 

clean:
		rm -rf StreamTriad *.o StreamTriad.cc
