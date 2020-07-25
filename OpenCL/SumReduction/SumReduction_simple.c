#include <stdio.h>
#include <sys/time.h>
#include "ezcl_lite.h"
#include "timer.h"

#include "SumReduction_kernel.inc"

int main(int argc, char *argv[]){

   size_t nsize = 200;
   cl_int iret=0;

   double *x = (double *)malloc(nsize*sizeof(double));

   for (int i = 0; i<nsize; i++){
     //x[i] = rand()*100.0;
     x[i] = 1.0;
   }   

   cl_context context;
   cl_command_queue command_queue;
   ezcl_devtype_init(CL_DEVICE_TYPE_GPU, &command_queue, &context);

   const char *defines = NULL;
   cl_program program = ezcl_create_program_wsource(context, defines, SumReduction_kernel_source);
   cl_kernel kernel_reduce_sum_stage1of2 = clCreateKernel(program, "reduce_sum_stage1of2_cl", &iret);
   cl_kernel kernel_reduce_sum_stage2of2 = clCreateKernel(program, "reduce_sum_stage2of2_cl", &iret);

   struct timespec tstart_cpu;
   cpu_timer_start(&tstart_cpu);

   size_t local_work_size = 128; 
   size_t global_work_size = ((nsize + local_work_size - 1) /local_work_size) * local_work_size;
   size_t nblocks     = global_work_size/local_work_size;

   cl_mem dev_x = clCreateBuffer(context, CL_MEM_READ_WRITE, nsize*sizeof(double), NULL, &iret);
   cl_mem dev_total_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, 1*sizeof(double), NULL, &iret);
   cl_mem dev_redscratch = clCreateBuffer(context, CL_MEM_READ_WRITE, nblocks*sizeof(double), NULL, &iret);

   clEnqueueWriteBuffer(command_queue, dev_x, CL_TRUE, 0, nsize*sizeof(cl_double), &x[0], 0, NULL, NULL);

   iret=clSetKernelArg(kernel_reduce_sum_stage1of2, 0, sizeof(cl_int), (void *)&nsize);
   iret=clSetKernelArg(kernel_reduce_sum_stage1of2, 1, sizeof(cl_mem), (void *)&dev_x);
   iret=clSetKernelArg(kernel_reduce_sum_stage1of2, 2, sizeof(cl_mem), (void *)&dev_total_sum);
   iret=clSetKernelArg(kernel_reduce_sum_stage1of2, 3, sizeof(cl_mem), (void *)&dev_redscratch);
   iret=clSetKernelArg(kernel_reduce_sum_stage1of2, 4, local_work_size*sizeof(cl_double), NULL);

   iret=clEnqueueNDRangeKernel(command_queue, kernel_reduce_sum_stage1of2, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

   if (nblocks > 1) {
      iret=clSetKernelArg(kernel_reduce_sum_stage2of2, 0, sizeof(cl_int), (void *)&nblocks);
      iret=clSetKernelArg(kernel_reduce_sum_stage2of2, 1, sizeof(cl_mem), (void *)&dev_total_sum);
      iret=clSetKernelArg(kernel_reduce_sum_stage2of2, 2, sizeof(cl_mem), (void *)&dev_redscratch);
      iret=clSetKernelArg(kernel_reduce_sum_stage2of2, 3, local_work_size*sizeof(cl_double), NULL);

      iret=clEnqueueNDRangeKernel(command_queue, kernel_reduce_sum_stage2of2, 1, NULL, &local_work_size, &local_work_size, 0, NULL, NULL);
   }

   double total_sum;

   iret=clEnqueueReadBuffer(command_queue, dev_total_sum, CL_TRUE, 0, 1*sizeof(cl_double), &total_sum, 0, NULL, NULL);

   //double gpu_sum_total = total_sum;
   printf("Result -- total sum %lf \n",total_sum);

   clReleaseMemObject(dev_x);
   clReleaseMemObject(dev_redscratch);
   clReleaseMemObject(dev_total_sum);

   clReleaseKernel(kernel_reduce_sum_stage1of2);
   clReleaseKernel(kernel_reduce_sum_stage2of2);
   clReleaseCommandQueue(command_queue);
   clReleaseContext(context);
   clReleaseProgram(program);

   free(x);

   //gpu_timers[STATE_TIMER_MASS_SUM] += (long)(cpu_timer_stop(tstart_cpu)*1.0e9);
}
