// CUDA kernel version of Stream Triad
__global__ void StreamTriad(
               const int n,
               const double scalar,
               const double *a,
               const double *b,
                     double *c)
{
   int i = blockIdx.x*blockDim.x+threadIdx.x;

   // Protect from going out-of-bounds
   if (i >= n) return;

   c[i] = a[i] + scalar*b[i];
}

#include <stdio.h>
#include <sys/time.h>
extern "C" {
   #include "timer.h"
}

#define NTIMES 16

int main(int argc, char *argv[]){
   struct timespec tkernel, ttotal;
   // initializing data and arrays
   int stream_array_size = 80000000;
   double scalar = 3.0, tkernel_sum = 0.0, ttotal_sum = 0.0;

   // allocate host memory and initialize
   double *a = (double *)malloc(stream_array_size*sizeof(double));
   double *b = (double *)malloc(stream_array_size*sizeof(double));
   double *c = (double *)malloc(stream_array_size*sizeof(double));

   for (int i=0; i<stream_array_size; i++) {
      a[i] = 1.0;
      b[i] = 2.0;
   }

   // allocate device memory. suffix of _d indicates a device pointer
   double *a_d, *b_d, *c_d;
   cudaMalloc(&a_d, stream_array_size*sizeof(double));
   cudaMalloc(&b_d, stream_array_size*sizeof(double));
   cudaMalloc(&c_d, stream_array_size*sizeof(double));

   // setting block size and padding total grid size to get even block sizes
   int blocksize = 512;
   int gridsize = (stream_array_size + blocksize - 1)/blocksize;

   for (int k=0; k<NTIMES; k++){
      cpu_timer_start(&ttotal);
      // copying array data from host to device
      cudaMemcpy(a_d, a, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(b_d, b, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);
      // cuda memcopy to device returns after buffer available, so synchronize to
      // get accurate timing for kernel only
      cudaDeviceSynchronize();

      cpu_timer_start(&tkernel);
      // launch stream triad kernel
      StreamTriad<<<gridsize, blocksize>>>(stream_array_size, scalar, a_d, b_d, c_d);
      // need to force completion to get timing
      cudaDeviceSynchronize();
      tkernel_sum += cpu_timer_stop(tkernel);

      // cuda memcpy from device to host blocks for completion so no need for synchronize
      cudaMemcpy(c, c_d, stream_array_size*sizeof(double), cudaMemcpyDeviceToHost);
      ttotal_sum += cpu_timer_stop(ttotal);
      // check results and print errors if found. limit to only 10 errors per iteration
      for (int i=0, icount=0; i<stream_array_size && icount < 10; i++){
         if (c[i] != 1.0 + 3.0*2.0) {
            printf("Error with result c[%d]=%lf on iter %d\n",i,c[i],k);
            icount++;
         }
      }
   }
   printf("Average runtime is %lf msecs data transfer is %lf msecs\n",
           tkernel_sum/NTIMES, (ttotal_sum - tkernel_sum)/NTIMES);

   cudaFree(a_d);
   cudaFree(b_d);
   cudaFree(c_d);

   free(a);
   free(b);
   free(c);
}
