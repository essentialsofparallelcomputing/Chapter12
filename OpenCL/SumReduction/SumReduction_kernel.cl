#define REDUCE_IN_TILE(operation, _spad_arr)                                    \
    for (int offset = ntX >> 1; offset > MIN_REDUCE_SYNC_SIZE; offset >>= 1)    \
    {                                                                           \
        if (tiX < offset)                                                       \
        {                                                                       \
            _spad_arr[tiX] = operation(_spad_arr[tiX], _spad_arr[tiX+offset]);  \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
    }                                                                           \
    if (tiX < MIN_REDUCE_SYNC_SIZE)                                             \
    {                                                                           \
        for (int offset = MIN_REDUCE_SYNC_SIZE; offset > 1; offset >>= 1)       \
        {                                                                       \
            _spad_arr[tiX] = operation(_spad_arr[tiX], _spad_arr[tiX+offset]);  \
            barrier(CLK_LOCAL_MEM_FENCE);                                       \
        }                                                                       \
        _spad_arr[tiX] = operation(_spad_arr[tiX], _spad_arr[tiX+1]);           \
    }

double SUM(double a, double b)
{
    return a + b; 
}

void reduction_sum_within_spad(__local  double  *spad)
{
   const unsigned int tiX  = get_local_id(0);
   const unsigned int ntX  = get_local_size(0);

   REDUCE_IN_TILE(SUM, spad);
}

__kernel void reduce_sum_stage1of2_cl(
                 const int      isize,      // 0  Total number of cells.
        __global const double  *array,      // 1 
        __global       double  *spadsum,    // 2 
        __global       double  *redscratch, // 3 
        __local        double  *spad)       // 4 
{
    const unsigned int giX  = get_global_id(0);
    const unsigned int tiX  = get_local_id(0);

    const unsigned int group_id = get_group_id(0);

    spad[tiX] = 0.0;
    if (giX < isize) {
      spad[tiX] = array[giX];
    }    

    barrier(CLK_LOCAL_MEM_FENCE);

    reduction_sum_within_spad(spad);

    //  Write the local value back to an array size of the number of groups
    if (tiX == 0){
      redscratch[group_id] = spad[0];
      (*spadsum) = spad[0];
    }    
}

__kernel void reduce_sum_stage2of2_cl(
                 const int    isize,
        __global       double *total_sum,
        __global       double *scratch,
        __local        double *spad)
{
   const unsigned int tiX  = get_local_id(0);
   const unsigned int ntX  = get_local_size(0);

   int giX = tiX; 

   spad[tiX] = 0.0; 

   // load the sum from redscratch
   if (tiX < isize) spad[tiX] = redscratch[giX];

   for (giX += ntX; giX < isize; giX += ntX) {
      spad[tiX] += redscratch[giX];
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   reduction_sum_within_spad(spad);

   if (tiX == 0) { 
     (*total_sum) = spad[0];
   }
}
