#include "CL/sycl.hpp"
extern "C" {
   #include "timer.h"
}  

namespace sycl = cl::sycl;

int main(int argc, char * argv[])
{
    struct timespec ttotal;
    double ttotal_sum = 0.0;

    size_t nsize = 10000;
    std::cout << "StreamTriad with " << nsize << " elements" << std::endl;

    // host data
    const double xval(1);
    const double yval(2);
    const double zval(2);
    std::vector<double> a(nsize,xval);
    std::vector<double> b(nsize,yval);
    std::vector<double> c(nsize,zval);

    cpu_timer_start(&ttotal);

    sycl::queue q(sycl::default_selector{});

    const double scalar = 3;

    sycl::buffer<double,1> dev_a { a.data(), sycl::range<1>(a.size()) };
    sycl::buffer<double,1> dev_b { b.data(), sycl::range<1>(b.size()) };
    sycl::buffer<double,1> dev_c { c.data(), sycl::range<1>(c.size()) };

    q.submit([&](sycl::handler& h) {

       auto a = dev_a.get_access<sycl::access::mode::read>(h);
       auto b = dev_b.get_access<sycl::access::mode::read>(h);
       auto c = dev_c.get_access<sycl::access::mode::read_write>(h);

       h.parallel_for<class StreamTriad>( sycl::range<1>{nsize}, [=] (sycl::id<1> it) {
           const int i = it[0];
           c[i] += a[i] + scalar * b[i];
       });
    });
    q.wait();

    ttotal_sum += cpu_timer_stop(ttotal);

    std::cout << "Program completed without error." << std::endl;
    std::cout << "Runtime is  " << ttotal_sum << " msecs " <<std::endl;
}
