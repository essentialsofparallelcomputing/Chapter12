#include <Kokkos_Core.hpp>

using namespace std;

int main (int argc, char *argv[])
{
   Kokkos::initialize(argc, argv);{

      Kokkos::Timer timer;
      double time1;

      double scalar = 3.0;
      size_t nsize = 1000000;
      Kokkos::View<double *> a( "a", nsize);
      Kokkos::View<double *> b( "b", nsize);
      Kokkos::View<double *> c( "c", nsize);

      cout << "StreamTriad with " << nsize << " elements" << endl;

      Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (int i) {
         a[i] = 1.0;
      });
      Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (int i) {
         b[i] = 2.0;
      });

      timer.reset();

      Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (const int i) {
         c[i] = a[i] + scalar * b[i];
      });

      time1 = timer.seconds();

      icount = 0;
      for (int i=0; i<nsize && icount < 10; i++){
         if (c[i] != 1.0 + 3.0*2.0) {
            cout << "Error with result c[" << i << "]=" << c[i] << endl;
            icount++;
         }
      }

      if (icount == 0) cout << "Program completed without error." << endl;
      cout << "Runtime is  " << time1*1000.0 << " msecs " << endl;

   }
   Kokkos::finalize();
   return 0;
}
