#include <Kokkos_Core.hpp>

using namespace std;

int main (int argc, char *argv[])
{
   Kokkos::Timer timer;
   double time1;

   size_t nsize = 1000000;
   double *c = new double[nsize];

   cout << "StreamTriad with " << nsize << " elements" << endl;

   Kokkos::initialize(argc, argv);{

      double *a = new double[nsize];
      double *b = new double[nsize];
      double scalar = 3.0;
      Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (int i) {
         a[i] = 1;
      });
      Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (int i) {
         b[i] = 2;
      });

      timer.reset();

      Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (const int i) {
         c[i] = a[i] + scalar * b[i];
      });

      time1 = timer.seconds();
   }
   Kokkos::finalize();

   for (int i=0, icount=0; i<nsize && icount < 10; i++){
      if (c[i] != 1.0 + 3.0*2.0) {
         cout << "Error with result c[" << i << "]=" << c[i] << endl;
         icount++;
      }
   }

   cout << "Program completed without error." << endl;
   cout << "Runtime is  " << time1*1000.0 << " msecs " << endl;

   return 0;
}
