#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timer.h"
#include "StreamTriad_kernel.inc"
#ifdef __APPLE_CC__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif
#include "ezcl_lite.h"

#define NTIMES 16

int main(int argc, char *argv[]){
   struct timespec tkernel, ttotal;
   int iret;
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

   cl_command_queue command_queue;
   cl_context context;
   iret = ezcl_devtype_init(CL_DEVICE_TYPE_GPU, &command_queue, &context);
   const char *defines = NULL;
   cl_program program  = ezcl_create_program_wsource(context, defines, StreamTriad_kernel_source);
   cl_kernel kernel_StreamTriad = clCreateKernel(program, "StreamTriad", &iret);

   // allocate device memory. suffix of _d indicates a device pointer
   size_t nsize = stream_array_size*sizeof(double);
   cl_mem a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, nsize, NULL, &iret);
   cl_mem b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, nsize, NULL, &iret);
   cl_mem c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, nsize, NULL, &iret);

   // setting block size and padding total grid size to get even block sizes
   size_t local_work_size = 512;
   size_t global_work_size = ( (stream_array_size + local_work_size - 1)/local_work_size ) * local_work_size;

   for (int k=0; k<NTIMES; k++){
      cpu_timer_start(&ttotal);
      // copying array data from host to device
      iret=clEnqueueWriteBuffer(command_queue, a_d, CL_FALSE, 0, nsize, &a[0], 0, NULL, NULL);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clEnqueueWriteBuffer");
      iret=clEnqueueWriteBuffer(command_queue, b_d, CL_TRUE, 0, nsize, &b[0], 0, NULL, NULL);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clEnqueueWriteBuffer");

      cpu_timer_start(&tkernel);
      // set stream triad kernel arguments
      iret=clSetKernelArg(kernel_StreamTriad, 0, sizeof(cl_int),    (void *)&stream_array_size);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clSetKernelArg");
      iret=clSetKernelArg(kernel_StreamTriad, 1, sizeof(cl_double), (void *)&scalar);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clSetKernelArg");
      iret=clSetKernelArg(kernel_StreamTriad, 2, sizeof(cl_mem),    (void *)&a_d);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clSetKernelArg");
      iret=clSetKernelArg(kernel_StreamTriad, 3, sizeof(cl_mem),    (void *)&b_d);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clSetKernelArg");
      iret=clSetKernelArg(kernel_StreamTriad, 4, sizeof(cl_mem),    (void *)&c_d);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clSetKernelArg");
      // call stream triad kernel
      clEnqueueNDRangeKernel(command_queue, kernel_StreamTriad, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clEnqueueNDRangeKernel");
      // need to force completion to get timing
      clEnqueueBarrier(command_queue);
      tkernel_sum += cpu_timer_stop(tkernel);

      iret=clEnqueueReadBuffer(command_queue, c_d, CL_TRUE, 0, nsize, c, 0, NULL, NULL);
      if (iret != CL_SUCCESS) ezcl_print_error(iret, "clEnqueueReadBuffer");
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

   clReleaseMemObject(a_d);
   clReleaseMemObject(b_d);
   clReleaseMemObject(c_d);

   clReleaseKernel(kernel_StreamTriad);
   clReleaseCommandQueue(command_queue);
   clReleaseContext(context);
   clReleaseProgram(program);

   free(a);
   free(b);
   free(c);
}
