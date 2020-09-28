#include <chrono>
#include "CL/sycl.hpp"

namespace Sycl = cl::sycl;
using namespace std;

int main(int argc, char * argv[])
{
   chrono::high_resolution_clock::time_point t1, t2;

   size_t nsize = 100000;
   cout << "StreamTriad with " << nsize << " elements" << endl;

   // host data
   vector<double> a(nsize,1.0);
   vector<double> b(nsize,2.0);
   vector<double> c(nsize,-1.0);

   t1 = chrono::high_resolution_clock::now();

   Sycl::queue Queue(sycl::cpu_selector{});

   const double scalar = 3.0;

   Sycl::buffer<double,1> dev_a { a.data(), Sycl::range<1>(a.size()) };
   Sycl::buffer<double,1> dev_b { b.data(), Sycl::range<1>(b.size()) };
   Sycl::buffer<double,1> dev_c { c.data(), Sycl::range<1>(c.size()) };

   Queue.submit([&](sycl::handler& CommandGroup) {

      auto a = dev_a.get_access<Sycl::access::mode::read>(CommandGroup);
      auto b = dev_b.get_access<Sycl::access::mode::read>(CommandGroup);
      auto c = dev_c.get_access<Sycl::access::mode::write>(CommandGroup);

      CommandGroup.parallel_for<class StreamTriad>( Sycl::range<1>{nsize}, [=] (Sycl::id<1> it) {
          c[it] = a[it] + scalar * b[it];
      });
   });
   Queue.wait();

   t2 = chrono::high_resolution_clock::now();

   int icount = 0;
   for (int i=0; i<nsize && icount < 10; i++){
      if (c[i] != 1.0 + 3.0*2.0) {
         cout << "Error with result c[" << i << "]=" << c[i] << endl;
         icount++;
      }
   }

   if (icount == 0) cout << "Program completed without error." << endl;
   double time1 = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
   cout << "Runtime is  " << time1*1000.0 << " msecs " << endl;
}
