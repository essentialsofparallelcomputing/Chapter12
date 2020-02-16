#define MIN_REDUCE_SYNC_SIZE warpSize

__device__ void reduction_sum_within_block(double  *spad)
{
   const unsigned int tiX  = threadIdx.x;
   const unsigned int ntX  = blockDim.x;

   for (int offset = ntX >> 1; offset > MIN_REDUCE_SYNC_SIZE; offset >>= 1) {
      if (tiX < offset) {
         spad[tiX] = spad[tiX] + spad[tiX+offset];
      }
      __syncthreads();
   }
   if (tiX < MIN_REDUCE_SYNC_SIZE) {
      for (int offset = MIN_REDUCE_SYNC_SIZE; offset > 1; offset >>= 1) {
         spad[tiX] = spad[tiX] + spad[tiX+offset];
         __syncthreads();
      }
      spad[tiX] = spad[tiX] + spad[tiX+1];
   }
}

__global__ void reduce_sum_stage1of2(
                 const int      isize,      // 0  Total number of cells.
                 const double  *array,      // 1
                       double  *blocksum,   // 2
                       double  *redscratch) // 3
{
    extern __shared__ double spad[];
    const unsigned int giX  = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int tiX  = threadIdx.x;

    const unsigned int group_id = blockIdx.x;

    spad[tiX] = 0.0;
    if (giX < isize) {
      spad[tiX] = array[giX];
    }

    __syncthreads();

    reduction_sum_within_block(spad);

    //  Write the local value back to an array size of the number of groups
    if (tiX == 0){
      redscratch[group_id] = spad[0];
      (*blocksum) = spad[0];
    }
}

__global__ void reduce_sum_stage2of2(
                 const int    isize,
                       double *total_sum,
                       double *redscratch)
{
   extern __shared__ double spad[];
   const unsigned int tiX  = threadIdx.x;
   const unsigned int ntX  = blockDim.x;

   int giX = tiX;

   spad[tiX] = 0.0;

   // load the sum from reduction scratch, redscratch
   if (tiX < isize) spad[tiX] = redscratch[giX];

   for (giX += ntX; giX < isize; giX += ntX) {
      spad[tiX] += redscratch[giX];
   }

   __syncthreads();

   reduction_sum_within_block(spad);

   if (tiX == 0) {
     (*total_sum) = spad[0];
   }
}

#include <stdio.h>
#include <sys/time.h>
extern "C" {
   #include "timer.h"
}

int main(int argc, char *argv[]){

   size_t nsize = 200;

   double *x = (double *)malloc(nsize*sizeof(double));

   for (int i = 0; i<nsize; i++){
     //x[i] = rand()*100.0;
     x[i] = 1.0;
   }

   struct timespec tstart_cpu;
   cpu_timer_start(&tstart_cpu);

   size_t blocksize = 128;
   size_t blocksizebytes = blocksize*sizeof(double);
   size_t global_work_size = ((nsize + blocksize - 1) /blocksize) * blocksize;
   size_t gridsize     = global_work_size/blocksize;

   double *dev_x, *dev_total_sum, *dev_redscratch;
   cudaMalloc(&dev_x, nsize*sizeof(double));
   cudaMalloc(&dev_total_sum, 1*sizeof(double));
   cudaMalloc(&dev_redscratch, gridsize*sizeof(double));

   cudaMemcpy(dev_x, x, nsize*sizeof(double), cudaMemcpyHostToDevice);

   reduce_sum_stage1of2<<<gridsize, blocksize, blocksizebytes>>>(nsize, dev_x, dev_total_sum, dev_redscratch);

   if (gridsize > 1) {
      reduce_sum_stage2of2<<<1, blocksize, blocksizebytes>>>(nsize, dev_total_sum, dev_redscratch);
   }

   double total_sum;
   cudaMemcpy(&total_sum, dev_total_sum, 1*sizeof(double), cudaMemcpyDeviceToHost);
   printf("Result -- total sum %lf \n",total_sum);

   cudaFree(dev_redscratch);
   cudaFree(dev_total_sum);
   cudaFree(dev_x);

   free(x);
}
