// OpenCL kernel version of stream triad
__kernel void StreamTriad(
               const int n,
               const double scalar,
      __global const double *a,
      __global const double *b,
      __global       double *c)
{
   int i = get_global_id(0);

   // Protect from going out-of-bounds
   if (i >= n) return;

   c[i] = a[i] + scalar*b[i];
}
