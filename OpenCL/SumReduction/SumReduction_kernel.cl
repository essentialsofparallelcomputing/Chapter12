#define REDUCE_IN_TILE(operation, _tile_arr)                                    \
    for (int offset = ntX >> 1; offset > MIN_REDUCE_SYNC_SIZE; offset >>= 1)    \
    {                                                                           \
        if (tiX < offset)                                                       \
        {                                                                       \
            _tile_arr[tiX] = operation(_tile_arr[tiX], _tile_arr[tiX+offset]);  \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
    }                                                                           \
    if (tiX < MIN_REDUCE_SYNC_SIZE)                                             \
    {                                                                           \
        for (int offset = MIN_REDUCE_SYNC_SIZE; offset > 1; offset >>= 1)       \
        {                                                                       \
            _tile_arr[tiX] = operation(_tile_arr[tiX], _tile_arr[tiX+offset]);  \
            barrier(CLK_LOCAL_MEM_FENCE);                                       \
        }                                                                       \
        _tile_arr[tiX] = operation(_tile_arr[tiX], _tile_arr[tiX+1]);           \
    }

double SUM(double a, double b)
{
    return a + b; 
}

void reduction_sum_within_tile(__local  double  *tile)
{
   const unsigned int tiX  = get_local_id(0);
   const unsigned int ntX  = get_local_size(0);

   REDUCE_IN_TILE(SUM, tile);
}

__kernel void reduce_sum_stage1of2_cl(
                 const int      isize,      // 0  Total number of cells.
        __global const double  *array,      // 1 
        __global       double  *blocksum,   // 2 
        __global       double  *redscratch, // 3 
        __local        double  *tile)       // 4 
{
    const unsigned int giX  = get_global_id(0);
    const unsigned int tiX  = get_local_id(0);

    const unsigned int group_id = get_group_id(0);

    tile[tiX] = 0.0;
    if (giX < isize) {
      tile[tiX] = array[giX];
    }    

    barrier(CLK_LOCAL_MEM_FENCE);

    reduction_sum_within_tile(tile);

    //  Write the local value back to an array size of the number of groups
    if (tiX == 0){
      redscratch[group_id] = tile[0];
      (*blocksum) = tile[0];
    }    
}

__kernel void reduce_sum_stage2of2_cl(
                 const int    isize,
        __global       double *mass_sum,
        __global       double *scratch,
        __local        double *tile)
{
   const unsigned int tiX  = get_local_id(0);
   const unsigned int ntX  = get_local_size(0);

   int giX = tiX; 

   tile[tiX] = 0.0; 

   // load the sum from scratch
   if (tiX < isize) tile[tiX] = scratch[giX];

   for (giX += ntX; giX < isize; giX += ntX) {
      tile[tiX] += scratch[giX];
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   reduction_sum_within_tile(tile);

   if (tiX == 0) { 
     (*mass_sum) = tile[0];
   }
}

