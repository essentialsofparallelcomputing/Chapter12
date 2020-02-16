#include "CL/sycl.hpp"
extern "C" {
   #include "timer.h"
}  

namespace Sycl = cl::sycl;
using namespace std;

int main(int argc, char * argv[])
{
    struct timespec ttotal;
    double ttotal_sum = 0.0;

    size_t nsize = 10000;
    cout << "StreamTriad with " << nsize << " elements" << endl;

    // host data
    vector<double> a(nsize,1.0);
    vector<double> b(nsize,2.0);
    vector<double> c(nsize,-1.0);

    cpu_timer_start(&ttotal);

    Sycl::queue Queue(sycl::default_selector{});

    const double scalar = 3.0;

    Sycl::buffer<double,1> dev_a { a.data(), Sycl::range<1>(a.size()) };
    Sycl::buffer<double,1> dev_b { b.data(), Sycl::range<1>(b.size()) };
    Sycl::buffer<double,1> dev_c { c.data(), Sycl::range<1>(c.size()) };

    Queue.submit([&](sycl::handler& Handler) {

       auto a = dev_a.get_access<Sycl::access::mode::read>(Handler);
       auto b = dev_b.get_access<Sycl::access::mode::read>(Handler);
       auto c = dev_c.get_access<Sycl::access::mode::write>(Handler);

       Handler.parallel_for<class StreamTriad>( Sycl::range<1>{nsize}, [=] (Sycl::id<1> it) {
           const int i = it[0];
           c[i] = a[i] + scalar * b[i];
       });
    });
    Queue.wait();

    for (int i=0, icount=0; i<nsize && icount < 10; i++){
       if (c[i] != 1.0 + 3.0*2.0) {
          cout << "Error with result c[" << i << "]=" << c[i] << endl;
          icount++;
       }
    }

    ttotal_sum += cpu_timer_stop(ttotal);

    cout << "Program completed without error." << endl;
    cout << "Runtime is  " << ttotal_sum << " msecs " << endl;
}
