CXX              = dpcpp
CXXFLAGS         = -std=c++17 -fsycl -O3

all: StreamTriad

StreamTriad: StreamTriad.o
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	-rm -f StreamTriad.o StreamTriad
