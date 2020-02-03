all: StreamTriad

#CFLAGS = -DDEVICE_DETECT_DEBUG=1
#OPENCL_LIB = <path>

%.inc : %.cl
	./embed_source.pl $^ > $@

StreamTriad.o: StreamTriad.c StreamTriad_kernel.inc

StreamTriad: StreamTriad.o timer.o ezcl_lite.o
	${CC} -o $@ $^ ${OPENCL_LIB} -lOpenCL

clean:
		rm -rf StreamTriad *.o StreamTriad_kernel.inc
