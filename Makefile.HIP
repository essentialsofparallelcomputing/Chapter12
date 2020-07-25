All: StreamTriad_hipify StreamTriad_hipifyclang StreamTriad_cmake

.PHONY: StreamTriad_hipify HIP/StreamTriad/hipifybuild/StreamTriad StreamTriad_hipifyclang HIP/StreamTriad/hipifyclang/StreamTriad StreamTriad_cmake HIP/StreamTriad/build/StreamTriad      

StreamTriad_hipify: HIP/StreamTriad/hipifybuild/StreamTriad

HIP/StreamTriad/hipifybuild/StreamTriad:
	cd HIP/StreamTriad && mkdir hipifybuild && cp * hipifybuild && cd hipifybuild && make -f Makefile.perl && ./StreamTriad

StreamTriad_hipifyclang: HIP/StreamTriad/hipifyclang/StreamTriad

HIP/StreamTriad/hipifyclang/StreamTriad:
	cd HIP/StreamTriad && mkdir hipifyclang && cp * hipifyclang && cd hipifyclang && make -f Makefile.clang && ./StreamTriad

StreamTriad_cmake: HIP/StreamTriad/build/StreamTriad

HIP/StreamTriad/build/StreamTriad:
	cd HIP/StreamTriad && mkdir build && cd build && cmake .. && make && ./StreamTriad

clean:
	cd HIP/StreamTriad && rm -rf hipifybuild hipifyclang build
