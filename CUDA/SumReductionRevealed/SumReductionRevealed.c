#include <stdio.h>
#include <stdlib.h>

#define MIN_REDUCE_SYNC_SIZE 32 //warpSize

int itree_level = 0;

void print_array(int gridsize, int blockDim, int isize, int offset, int *array)
{
   int icount = 0;
   int isum = 0;
   for (int blockIdx=0; blockIdx<gridsize; blockIdx++){
      for (int tiX=0; tiX<blockDim; tiX++){
         int giX = tiX + blockIdx*blockDim;
         if (tiX < offset && giX < isize) {
            printf("%3d ",array[giX]);
            isum += array[giX];
            icount++;
         }
      } // tiX
      printf("   ");
   } // block
   printf("\nSum is %d. Data count is reduced to %d\n",isum,icount);
}

void reduction_sum_within_block(int gridsize, int blockDim, int isize, int *array)
{
   const unsigned int ntX  = blockDim;

   print_array(gridsize, blockDim, isize, ntX, array);

   for (int offset = ntX >> 1; offset > MIN_REDUCE_SYNC_SIZE; offset >>= 1) {
      if (offset >= isize) continue;
      itree_level++;
      printf("\n ====== ITREE_LEVEL %d offset %d ntX is %d MIN_REDUCE_SYNC_SIZE %d ====\n",itree_level,offset,ntX,MIN_REDUCE_SYNC_SIZE);
      for (int blockstart=0; blockstart<gridsize; blockstart++){

         for (int tiX=0; tiX<blockDim; tiX++){
            int giX = tiX + blockstart*blockDim;
            if (tiX < offset){
               array[giX]+=array[giX+offset];
            }
         }
      }

      print_array(gridsize, blockDim, isize, offset, array);
      printf("Sync threads when larger than warp\n");

   } //offset

   for (int offset = MIN_REDUCE_SYNC_SIZE; offset > 1; offset >>= 1) {
      if (offset >= isize) continue;
      itree_level++;
      printf("\n ====== ITREE_LEVEL %d offset %d ntX is %d MIN_REDUCE_SYNC_SIZE %d ====\n",itree_level,offset,ntX,MIN_REDUCE_SYNC_SIZE);

      for (int blockstart=0; blockstart<gridsize; blockstart++){
         for (int tiX=0; tiX<blockDim; tiX++){
            if (tiX < MIN_REDUCE_SYNC_SIZE) {
               int giX = tiX + blockstart*blockDim;
               if (tiX < offset) {
                  array[giX]+=array[giX+offset];
               }
            }
         }
      }
      printf("Sync threads when smaller than warp\n");
      print_array(gridsize, blockDim, isize, offset, array);
   }

   int offset = 1;
   itree_level++;
   printf("\n ====== ITREE_LEVEL %d offset %d ntX is %d MIN_REDUCE_SYNC_SIZE %d ====\n",itree_level,offset,ntX,MIN_REDUCE_SYNC_SIZE);

   for (int blockstart=0; blockstart<gridsize; blockstart++){
      int tiX = 0;
      int giX = tiX + blockstart*blockDim;
      array[giX]+=array[giX+offset];
   }
   print_array(gridsize, blockDim, isize, offset, array);
   printf("\nFinished reduction sum within thread block\n\n");
}

void reduce_sum_stage1of2_revealed(int gridsize, int blockDim, int isize, int *array)
{
   printf("Test data at start. First row is thread number. Second row is the data.\n");
   printf("The end of the array after 199 should not have data. Spaces are output between thread blocks\n");

   // printing out thread numbers
   for (int blockIdx=0; blockIdx<gridsize; blockIdx++){
      for (int tiX=0; tiX<blockDim; tiX++){
         int giX = tiX + blockIdx*blockDim;

         printf("%3d ",giX);
      } // tiX
      printf("   ");
   } // block
   printf("\n");

   print_array(gridsize, blockDim, isize, blockDim, array);

   printf("End of test data\n\n");

   printf("SYNCTHREADS after all values are in shared memory block\n");
   //__syncthreads();

   reduction_sum_within_block(gridsize,blockDim,isize,array);

   for (int blockstart=0; blockstart<gridsize; blockstart++){
      int giX = blockstart*blockDim;
      printf("Sum by block for block %d is %d\n",blockstart,array[giX]);
      if (gridsize == 1) {
         printf("\n Can skip second pass and the sum of the array is the block sum %d\n",array[0]);
      }
   }

   printf("\nEnd of first pass\n\n");
}

void reduce_sum_stage2of2_revealed(int gridsize, int blockDim, int isize, int *array)
{
   const unsigned int ntX  = blockDim;

   for (int tiX=0; tiX<isize; tiX++){
      int giX = tiX * blockDim;
      array[tiX] = array[giX];
   }

   for (int tiX=0; tiX<blockDim && tiX<isize; tiX++){
      int giX = tiX;
      for (giX += ntX; giX < isize; giX += ntX) {
         array[tiX] += array[giX];
      }
   }
   printf("Synchronization in second pass after loading data\n");

   reduction_sum_within_block(1,blockDim,isize,array);

   printf("Synchronization in second pass after reduction sum\n");
}


int main(int argc, char *argv[]){

   size_t nsize = 200;

   size_t blocksize = 128;
   size_t blocksizebytes = blocksize*sizeof(double);
   size_t global_work_size = ((nsize + blocksize - 1) /blocksize) * blocksize;
   size_t gridsize     = global_work_size/blocksize;

   int *array = (int *)malloc(nsize*sizeof(int));
   int isum = 0;
   for (int i = 0; i<nsize; i++){
     array[i]=i;
     isum += i;
   }
   printf("\nSum of test data should be %d\n\n",isum);

   printf("Calling first pass with gridsize %d blocksize %d blocksizebytes %d\n\n",
          gridsize,blocksize,blocksizebytes);

   reduce_sum_stage1of2_revealed(gridsize,blocksize,nsize,array);//<<<gridsize, blocksize, blocksizebytes>>>(nsize, dev_x, dev_total_sum, dev_redscratch);

   if (gridsize > 1) {
      reduce_sum_stage2of2_revealed(1, blocksize, gridsize, array);//<<<gridsize, blocksize, blocksizebytes>>>(nsize, dev_total_sum, dev_redscratch);
   }

   printf("Result -- total sum %d \n",array[0]);

   free(array);
}
