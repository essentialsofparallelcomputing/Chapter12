CXX              = dpcpp
CXXFLAGS         = -std=c++17 -fsycl -O3
EXTFLAGS         = -fsycl-unnamed-lambda

all: StreamTriad

StreamTriad: StreamTriad.cc timer.o
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f StreamTriad.o StreamTriad
