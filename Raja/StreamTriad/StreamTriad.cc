#include <chrono>
#include "RAJA/RAJA.hpp"

using namespace std;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
   chrono::high_resolution_clock::time_point t1, t2;
   cout << "Running Raja Stream Triad\n";

   const int nsize = 1000000;

// Allocate and initialize vector data.
   double scalar = 3.0;
   double* a = new double[nsize];
   double* b = new double[nsize];
   double* c = new double[nsize];
  
   for (int i = 0; i < nsize; i++) {
     a[i] = 1.0;
     b[i] = 2.0;
   }

   t1 = chrono::high_resolution_clock::now();

   RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, nsize), [=] (int i) {
     c[i] = a[i] + scalar * b[i];
   });

   t2 = chrono::high_resolution_clock::now();

   // check results and print errors if found. limit to only 10 errors per iteration
   int icount = 0;
   for (int i=0; i<nsize && icount < 10; i++){
      if (c[i] != 1.0 + 3.0*2.0) {
         printf("Error with result c[%d]=%lf\n",i,c[i]);
         icount++;
      }
   }

   if (icount == 0) cout << "Program completed without error." << endl;
   double time1 = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
   cout << "Runtime is  " << time1*1000.0 << " msecs " << endl;
}
