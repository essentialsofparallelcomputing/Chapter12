/*
 *  Copyright (c) 2011-2019, Triad National Security, LLC.
 *  All rights Reserved.
 *
 *  CLAMR -- LA-CC-11-094
 *
 *  Copyright 2011-2019. Triad National Security, LLC. This software was produced 
 *  under U.S. Government contract 89233218CNA000001 for Los Alamos National 
 *  Laboratory (LANL), which is operated by Triad National Security, LLC 
 *  for the U.S. Department of Energy. The U.S. Government has rights to use, 
 *  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
 *  TRIAD NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
 *  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
 *  to produce derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Triad National Security, LLC, Los Alamos 
 *       National Laboratory, LANL, the U.S. Government, nor the names of its 
 *       contributors may be used to endorse or promote products derived from 
 *       this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE TRIAD NATIONAL SECURITY, LLC AND 
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT 
 *  NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL TRIAD NATIONAL
 *  SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *  
 *  CLAMR -- LA-CC-11-094
 *  This research code is being developed as part of the 
 *  2011 X Division Summer Workshop for the express purpose
 *  of a collaborative code for development of ideas in
 *  the implementation of AMR codes for Exascale platforms
 *  
 *  AMR implementation of the Wave code previously developed
 *  as a demonstration code for regular grids on Exascale platforms
 *  as part of the Supercomputing Challenge and Los Alamos 
 *  National Laboratory
 *  
 *  Authors: Bob Robey       XCP-2   brobey@lanl.gov
 *           Neal Davis              davis68@lanl.gov, davis68@illinois.edu
 *           David Nicholaeff        dnic@lanl.gov, mtrxknight@aol.com
 *           Dennis Trujillo         dptrujillo@lanl.gov, dptru10@gmail.com
 * 
 */
#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef DEVICE_DETECT_DEBUG
#define DEVICE_DETECT_DEBUG 0
#endif

//#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/queue.h>
//#include <sys/stat.h>
#include <execinfo.h>

#include "ezcl_lite.h"

static cl_device_id    *devices;
static int compute_device = 0;

int ezcl_device_double_support(cl_device_id device);
void ezcl_device_info(cl_device_id device);

cl_int ezcl_devtype_init_p(cl_device_type device_type, cl_command_queue *command_queue, cl_context *context,
                           const char *file, int line){
   cl_device_id device;
   int ierr;
   cl_uint nDevices_selected=0;

   cl_platform_id  *platforms;
   cl_uint nPlatforms = 0;
   cl_uint          nDevices = 0;
   cl_int platform_selected = -1;
   cl_platform_id   platform = NULL;
   int *device_appropriate;
   int device_selected = -99;

   // Get the number of platforms first, then allocate and get the platform
   ierr = clGetPlatformIDs(0, NULL, &nPlatforms);
   if (ierr != CL_SUCCESS){
      printf("EZCL_DEVTYPE_INIT: Error with clGetDeviceIDs call in file %s at line %d\n", file, line);
      if (ierr == CL_INVALID_VALUE){
         printf("Invalid value in clGetPlatformID call\n");
      }
      exit(-1);
   }
   if (nPlatforms == 0) {
      printf("EZCL_DEVTYPE_INIT: Error -- No opencl platforms detected in file %s at line %d\n", file, line);
      exit(-1);
   }
   if (DEVICE_DETECT_DEBUG){
      printf("\n\nEZCL_DEVTYPE_INIT: %d opencl platform(s) detected\n",nPlatforms);
   }

   platforms = (cl_platform_id *)malloc(nPlatforms*sizeof(cl_platform_id));
   //ezcl_malloc_memory_add(platforms, "PLATFORMS", nPlatforms*sizeof(cl_platform_id));

   ierr = clGetPlatformIDs(nPlatforms, platforms, NULL);
   if (ierr != CL_SUCCESS){
      printf("EZCL_DEVTYPE_INIT: Error with clGetPlatformIDs call in file %s at line %d\n", file, line);
      if (ierr == CL_INVALID_VALUE){
         printf("Invalid value in clGetPlatformID call\n");
      }
   }

   if (DEVICE_DETECT_DEBUG){
      char info[1024];
      for (uint iplatform=0; iplatform<nPlatforms; iplatform++){
         printf("  Platform %d:\n",iplatform+1);

         //clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_PROFILE,   1024L,info,0);
         //printf("    CL_PLATFORM_PROFILE    : %s\n",info);

         clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_VERSION,   1024L,info,0);
         printf("    CL_PLATFORM_VERSION    : %s\n",info);

         clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_NAME,      1024L,info,0);
         printf("    CL_PLATFORM_NAME       : %s\n",info);

         clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_VENDOR,    1024L,info,0);
         printf("    CL_PLATFORM_VENDOR     : %s\n",info);

         //clGetPlatformInfo(platforms[iplatform],CL_PLATFORM_EXTENSIONS,1024L,info,0);
         //printf("    CL_PLATFORM_EXTENSIONS : %s\n",info);
      }
      printf("\n");
   }

   // Get the number of devices, allocate, and get the devices
   for (uint iplatform=0; iplatform<nPlatforms; iplatform++){
      ierr = clGetDeviceIDs(platforms[iplatform],device_type,0,NULL,&nDevices);
      if (ierr == CL_DEVICE_NOT_FOUND) {
         if (DEVICE_DETECT_DEBUG) {
           printf("Warning: Device of requested type not found for platform %d in clGetDeviceID call\n",iplatform);
         }
         continue;
      }
      if (ierr != CL_SUCCESS) ezcl_print_error(ierr, "clGetDeviceIDs");
      if (DEVICE_DETECT_DEBUG){
         printf("EZCL_DEVTYPE_INIT: %d opencl devices(s) detected\n",nDevices);
      }
      platform_selected = iplatform;
      platform = platforms[iplatform];
      nDevices_selected = nDevices;
   }

   if (platform_selected == -1){
      printf("Warning: Device of requested type not found in clGetDeviceID call\n");
      //ezcl_malloc_memory_delete(platforms);
      return(EZCL_NODEVICE);
   }

   nDevices = nDevices_selected;

   devices = (cl_device_id *)malloc(nDevices*sizeof(cl_device_id));
   device_appropriate = malloc(nDevices*sizeof(int));

   ierr = clGetDeviceIDs(platforms[platform_selected],device_type,nDevices,devices,NULL);
   if (ierr != CL_SUCCESS) ezcl_print_error(ierr, "clGetDeviceIDs");

// Not working quite right yet -- trying to skip non-double capable devices

   int idevice_appropriate = 0;
   for (uint idevice=0; idevice<nDevices; idevice++){
      device_appropriate[idevice] = ezcl_device_double_support(devices[idevice]);;
      if (device_appropriate[idevice] == 1){
         if (device_selected == -99) device_selected = idevice;
         devices[idevice_appropriate] = devices[idevice];
         idevice_appropriate++;
      }
      if (DEVICE_DETECT_DEBUG){
         printf(  "  Device %d:\n", idevice+1);
         ezcl_device_info(devices[idevice]);
      }
   }
   nDevices = idevice_appropriate; 

   if (DEVICE_DETECT_DEBUG) {
      printf("Device selected is %d number of appropriate devices %d\n",device_selected, nDevices);
   }

   cl_context_properties context_properties[3]=
   {
     CL_CONTEXT_PLATFORM,
     (cl_context_properties)platform,
     0 // 0 terminates list
 };

   *context = clCreateContext(context_properties, nDevices, devices, NULL, NULL, &ierr);
   if (ierr == CL_INVALID_VALUE){
      printf("Invalid value in clCreateContext call\n");
      if (devices == NULL) printf("Devices is NULL\n");
   }
   if (ierr != CL_SUCCESS) ezcl_print_error(ierr, "clCreateContext");
   if (DEVICE_DETECT_DEBUG){
      if (*context != NULL){
         if(device_type & CL_DEVICE_TYPE_CPU )
            printf("EZCL_DEVTYPE_INIT: CPU device context created\n");
         else if (device_type & CL_DEVICE_TYPE_GPU){
            printf("EZCL_DEVTYPE_INIT: GPU device context created\n");
         }
         else if (device_type & CL_DEVICE_TYPE_ACCELERATOR){
            printf("EZCL_DEVTYPE_INIT: ACCELERATOR device context created\n");
         }
         else if (device_type & CL_DEVICE_TYPE_DEFAULT){
            printf("EZCL_DEVTYPE_INIT: Default device context created\n");
       }
      } else {
         if (DEBUG == 2) printf("EZCL_DEVTYPE_INIT: No device of type specified found\n");
      }
   }

   clGetContextInfo(*context, CL_CONTEXT_DEVICES, sizeof(device), &device, NULL);
   if (*context == NULL){
      printf("EZCL_DEVTYPE_INIT: Failed to find device and setup context in file %s at line %d\n", file, line);
      exit(-1); /* No device is available, something is wrong */
   }
   if (DEVICE_DETECT_DEBUG == 2){
      ezcl_device_info(device);
   }

   char info[1024];

   clGetDeviceInfo(devices[device_selected], CL_DEVICE_VENDOR, sizeof(info), &info, NULL);
   if (DEVICE_DETECT_DEBUG) printf("DEVICE VENDOR is %s\n",info);
   if (! strncmp(info,"NVIDIA",(size_t)6) ) compute_device = COMPUTE_DEVICE_NVIDIA;
   if (! strncmp(info,"Advanced Micro Devices",(size_t)6) ) compute_device = COMPUTE_DEVICE_AMD;
   if (! strncmp(info,"AMD",(size_t)3) ) compute_device = COMPUTE_DEVICE_AMD;
   if (! strncmp(info,"Intel",(size_t)5) ) compute_device = COMPUTE_DEVICE_INTEL;
   if (DEVICE_DETECT_DEBUG) {
      printf("DEBUG -- device vendor is |%s|, compute_device %d\n",info,compute_device);
   }

   int mype = 0;

#ifdef XXX
//#ifndef __APPLE_CC__
   if (numpe_node > (int)nDevices) {
      printf("%d:EZCL_DEVTYPE_INIT: Error -- not enough GPUs for mpi ranks. nDevices %d numpe_node %d\n",
             mype,nDevices,numpe_node);
      exit(-1); /* Not enough devices for mpi ranks */
   }
//#endif
#endif

   // Control flags
   static struct {
      unsigned int timing     : 1;
   } ezcl_flags={1};

   cl_command_queue_properties queueProps = ezcl_flags.timing ? CL_QUEUE_PROFILING_ENABLE : 0;
   *command_queue = clCreateCommandQueue(*context, devices[mype%nDevices], queueProps, &ierr);
   if (ierr != CL_SUCCESS) ezcl_print_error(ierr, "clCreateCommandQueue");

   return(EZCL_SUCCESS);
}

int ezcl_device_double_support(cl_device_id device){
   int have_double = 0;
   char info[1024];

   clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(info), &info, NULL);

   if (!(strstr(info,"cl_khr_fp64") == NULL)){
     if (DEVICE_DETECT_DEBUG){
        printf(  "    Device has double : %s\n\n", strstr(info,"cl_khr_fp64"));
     }
     have_double = 1;
   }

   return(have_double);
}

void ezcl_device_info(cl_device_id device){
   char info[1024];
   cl_bool iflag;
   cl_uint inum;
   size_t isize;
   cl_ulong ilong;
   cl_device_type device_type;
   cl_command_queue_properties iprop;

   clGetDeviceInfo(device,CL_DEVICE_TYPE,sizeof(device_type),&device_type,0);
   if( device_type & CL_DEVICE_TYPE_CPU )
      printf("    CL_DEVICE_TYPE                       : %s\n", "CL_DEVICE_TYPE_CPU");
   if( device_type & CL_DEVICE_TYPE_GPU )
      printf("    CL_DEVICE_TYPE                       : %s\n", "CL_DEVICE_TYPE_GPU");
   if( device_type & CL_DEVICE_TYPE_ACCELERATOR )
      printf("    CL_DEVICE_TYPE                       : %s\n", "CL_DEVICE_TYPE_ACCELERATOR");
   if( device_type & CL_DEVICE_TYPE_DEFAULT )
      printf("    CL_DEVICE_TYPE                       : %s\n", "CL_DEVICE_TYPE_DEFAULT");
               
   clGetDeviceInfo(device,CL_DEVICE_AVAILABLE,sizeof(iflag),&iflag,0);
   if (iflag == CL_TRUE) {
      printf(  "    CL_DEVICE_AVAILABLE                  : TRUE\n");
   } else {
      printf(  "    CL_DEVICE_AVAILABLE                  : FALSE\n");
   }
   
   clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(info), &info, NULL);
   printf(  "    CL_DEVICE_VENDOR                     : %s\n", info);
   
   clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), &info, NULL);
   printf(  "    CL_DEVICE_NAME                       : %s\n", info);
   
   clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(info), &info, NULL);
   printf(  "    CL_DRIVER_VERSION                    : %s\n", info);
   
   clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(info), &info, NULL);
   printf(  "    CL_DEVICE_VERSION                    : %s\n", info);
   
   clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_MAX_COMPUTE_UNITS          : %d\n", inum);
   
   clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS   : %d\n", inum);
   
   size_t *item_sizes = (size_t *)malloc(inum*sizeof(size_t));
   clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(item_sizes),item_sizes,0);
   printf(  "    CL_DEVICE_MAX_WORK_ITEM_SIZES        : %ld %ld %ld\n",
         item_sizes[0], item_sizes[1], item_sizes[2]);
   free(item_sizes);
   
   clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(isize),&isize,0);
   printf(  "    CL_DEVICE_MAX_WORK_GROUP_SIZE        : %ld\n", isize);
   
   clGetDeviceInfo(device,CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_MAX_CLOCK_FREQUENCY        : %d\n", inum);
   
   clGetDeviceInfo(device,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_MAX_MEM_ALLOC_SIZE         : %d\n", inum);
   
#ifdef __APPLE_CC__
   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(ilong),&ilong,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_SIZE            : %llu\n", ilong);
   
   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(ilong),&ilong,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE      : %llu\n", ilong);
#else
   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(ilong),&ilong,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_SIZE            : %lu\n", ilong);

   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(ilong),&ilong,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE      : %lu\n", ilong);
#endif
   clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE  : %d\n", inum);
   
   clGetDeviceInfo(device,CL_DEVICE_MAX_CONSTANT_ARGS,sizeof(inum),&inum,0);
   printf(  "    CL_DEVICE_GLOBAL_MAX_CONSTANT_ARGS   : %d\n", inum);
   
   clGetDeviceInfo(device,CL_DEVICE_ERROR_CORRECTION_SUPPORT,sizeof(iflag),&iflag,0);
   if (iflag == CL_TRUE) {
      printf(  "    CL_DEVICE_ERROR_CORRECTION_SUPPORT   : TRUE\n");
   } else {
      printf(  "    CL_DEVICE_ERROR_CORRECTION_SUPPORT   : FALSE\n");
   }
   
   clGetDeviceInfo(device,CL_DEVICE_PROFILING_TIMER_RESOLUTION,sizeof(isize),&isize,0);
   printf(  "    CL_DEVICE_PROFILING_TIMER_RESOLUTION : %ld nanosecs\n", isize);
   
   clGetDeviceInfo(device,CL_DEVICE_QUEUE_PROPERTIES,sizeof(iprop),&iprop,0);
   if (iprop & CL_QUEUE_PROFILING_ENABLE) {
      printf(  "    CL_DEVICE_QUEUE PROFILING            : AVAILABLE\n");
   } else {
      printf(  "    CL_DEVICE_QUEUE PROFILING            : NOT AVAILABLE\n");
   }
   
   clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(info), &info, NULL);
   printf(  "    CL_DEVICE_EXTENSIONS                 : %s\n\n", info);
   
}

char * create_compile_string(void)
{
   char * CompileString = calloc(200, sizeof(char));

#ifdef HAVE_CL_DOUBLE
   strcat(CompileString,"-DHAVE_CL_DOUBLE");
#else
   strcat(CompileString,"-DNO_CL_DOUBLE -cl-single-precision-constant");
#endif

   if (compute_device == COMPUTE_DEVICE_NVIDIA) {
      strcat(CompileString," -DIS_NVIDIA -DMIN_REDUCE_SYNC_SIZE=32 -DN_BITS_WARP_LENGTH=5");
   }
   else if (compute_device == COMPUTE_DEVICE_AMD) {
      strcat(CompileString, " -DIS_AMD -DMIN_REDUCE_SYNC_SIZE=64 -DN_BITS_WARP_LENGTH=6");
   }
   else if (compute_device == COMPUTE_DEVICE_INTEL) {
      strcat(CompileString, " -DIS_AMD -DMIN_REDUCE_SYNC_SIZE=64 -DN_BITS_WARP_LENGTH=6");
   }
   else
   {
        printf("EZCL_CREATE_KERNEL: Not supporting anything other than AMD or NVIDIA at the moment\n");
        exit(EXIT_FAILURE);
   }

   return(CompileString);
}

cl_program ezcl_create_program_wsource_p(cl_context context, const char *defines, const char *source, const char *file, const int line){
   cl_int ierr;
   size_t nReportSize;
   char *BuildReport;
   char *filename_copy;
   
   cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &ierr);
   if (ierr != CL_SUCCESS){
      printf("EZCL_CREATE_PROGRAM_WSOURCE: clCreateProgramWithSource returned an error %d in file %s at line %d\n",ierr, file, line);
      switch (ierr){
      case CL_INVALID_CONTEXT:
       printf("Invalid context in clCreatProgramWithSource\n");
       break;
      case CL_INVALID_VALUE:
        printf("Invalid value in clCreateProgramWithSource\n");
        break;
      case CL_OUT_OF_HOST_MEMORY:
        printf("Out of host memory in clCreateProgramWithSource\n");
        break;
     }
   }

   char * CompileString = create_compile_string();

   //printf("DEBUG file %s line %d CompileString %s\n",__FILE__,__LINE__,CompileString);
   ierr = clBuildProgram(program, 0, NULL, CompileString, NULL, NULL);

   free(CompileString);

   if (ierr != CL_SUCCESS){
      printf("EZCL_CREATE_PROGRAM_WSOURCE: clBuildProgram returned an error %d in file %s at line %d\n",ierr, file, line);
        switch (ierr){
        case CL_INVALID_PROGRAM:
          printf("Invalid program in clBuildProgram\n");
          break;
        case CL_INVALID_VALUE:
          printf("Invalid value in clBuildProgram\n");
          break;
        case CL_INVALID_DEVICE:
          printf("Invalid device in clBuildProgram\n");
          break;
        case CL_INVALID_BUILD_OPTIONS:
          printf("Invalid build options in clBuildProgram\n");
          break;
        case CL_INVALID_OPERATION:
          printf("Invalid operation in clBuildProgram\n");
          break;
        case CL_COMPILER_NOT_AVAILABLE:
          printf("CL compiler not available in clBuildProgram\n");
          break;
        case CL_BUILD_PROGRAM_FAILURE:
          printf("Build program failure in clBuildProgram\n");
          ierr = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0L, NULL, &nReportSize);
          if (ierr != CL_SUCCESS) {
            switch (ierr){
               case CL_INVALID_DEVICE:
                  printf("Invalid device in clProgramBuildInfo\n");
                  break;
               case CL_INVALID_VALUE:
                  printf("Invalid value in clProgramBuildInfo\n");
                  break;
               case CL_INVALID_PROGRAM:
                  printf("Invalid program in clProgramBuildInfo\n");
                  break;
               }
            }
                 
            BuildReport = (char *)malloc(nReportSize);
                 
            ierr = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, nReportSize, BuildReport, NULL);
            if (ierr != CL_SUCCESS) {
               switch (ierr){
                  case CL_INVALID_DEVICE:
                     printf("Invalid device in clProgramBuildInfo\n");
                     break;
                  case CL_INVALID_VALUE:
                     printf("Invalid value in clProgramBuildInfo\n");
                     break;
                  case CL_INVALID_PROGRAM:
                     printf("Invalid program in clProgramBuildInfo\n");
                     break;
               }
            }
              
            printf("EZCL_CREATE_KERNEL: Build Log: %s\n",BuildReport);
            free(BuildReport);
            exit(-1);
            break;
        case CL_OUT_OF_HOST_MEMORY:
          printf("Out of host memory in clBuildProgram\n");
          break;
      }
   } else {
      if (DEBUG)
         printf("EZCL_CREATE_PROGRAM_WSOURCE: Build is SUCCESSFUL with no errors\n");

      if (compute_device == COMPUTE_DEVICE_INTEL) {
         ierr = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0L, NULL, &nReportSize);
         BuildReport = (char *)malloc(nReportSize);
         ierr = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, nReportSize, BuildReport, NULL);
         printf("EZCL_CREATE_PROGRAM_WSOURCE: Build Log: %s\n",BuildReport);
         free(BuildReport);
      }
   }

   return(program);
}

void ezcl_print_error_p(const int ierr, const char *routine, const char *cl_routine, const char *file, const int line)
{
   switch (ierr){
      case CL_DEVICE_NOT_FOUND:                //#define CL_DEVICE_NOT_FOUND                 -1
         printf("\nERROR: %s -- Device not found in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_DEVICE_NOT_AVAILABLE:            //#define CL_DEVICE_NOT_AVAILABLE             -2
         printf("\nERROR: %s -- Device not available in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_COMPILER_NOT_AVAILABLE:          //#define CL_COMPILER_NOT_AVAILABLE           -3
         printf("\nERROR: %s -- CL compiler not available failure in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:   //#define CL_MEM_OBJECT_ALLOCATION_FAILURE    -4
         printf("\nERROR: %s -- Mem object allocation failure in %s at line %d in file %s\n", routine, cl_routine, line, file);
         if (compute_device == COMPUTE_DEVICE_NVIDIA) {
            system("nvidia-smi -q -d MEMORY");
         }
         break;
      case CL_OUT_OF_RESOURCES:                //#define CL_OUT_OF_RESOURCES                 -5
         printf("\nERROR: %s -- Out of resources in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_OUT_OF_HOST_MEMORY:              //#define CL_OUT_OF_HOST_MEMORY               -6
         printf("\nERROR: %s -- Out of host memory in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:    //#define CL_PROFILING_INFO_NOT_AVAILABLE     -7
         printf("\nERROR: %s -- Profiling info not available in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_MEM_COPY_OVERLAP:                //#define CL_MEM_COPY_OVERLAP                 -8
         printf("\nERROR: %s -- Mem copy overlap in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_IMAGE_FORMAT_MISMATCH:           //#define CL_IMAGE_FORMAT_MISMATCH            -9
         printf("\nERROR: %s -- Image format mismatch in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:      //#define CL_IMAGE_FORMAT_NOT_SUPPORTED      -10
         printf("\nERROR: %s -- Image format not supported in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_BUILD_PROGRAM_FAILURE:           //#define CL_BUILD_PROGRAM_FAILURE           -11
         printf("\nERROR: %s -- Build program failure in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_MAP_FAILURE:                     //#define CL_MAP_FAILURE                     -12
         printf("\nERROR: %s -- Map failure in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
#ifdef CL_MISALIGNED_SUB_BUFFER_OFFSET
      case CL_MISALIGNED_SUB_BUFFER_OFFSET:    //#define CL_MISALIGNED_SUB_BUFFER_OFFSET    -13
         printf("\nERROR: %s -- Misaligned sub buffer offset in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
#endif
#ifdef CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
      case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:  //#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST    -14
         printf("\nERROR: %s -- Error for events in wait list in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
#endif
#ifdef CL_COMPILE_PROGRAM_FAILURE
      case CL_COMPILE_PROGRAM_FAILURE:         //#define CL_COMPILE_PROGRAM_FAILURE         -15
         printf("\nERROR: %s -- Compile program failure in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
#endif
#ifdef CL_LINKER_NOT_AVAILABLE
      case CL_LINKER_NOT_AVAILABLE:            //#define CL_LINKER_NOT_AVAILABLE            -16
         printf("\nERROR: %s -- Linker not available in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
#endif
#ifdef CL_LINK_PROGRAM_FAILURE
      case CL_LINK_PROGRAM_FAILURE:            //#define CL_LINK_PROGRAM_FAILURE            -17
         printf("\nERROR: %s -- Link program failure in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
#endif
#ifdef CL_DEVICE_PARTITION_FAILED
      case CL_DEVICE_PARTITION_FAILED:         //#define CL_DEVICE_PARTITION_FAILED         -18
         printf("\nERROR: %s -- Device partition failed in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
#endif
#ifdef CL_KERNEL_ARG_INFO_NOT_AVAILABLE
      case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:   //#define CL_KERNEL_ARG_INFO_NOT_AVAILABLE   -19
         printf("\nERROR: %s -- Kernel arg info not available in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
#endif

      case CL_INVALID_VALUE:                   //#define CL_INVALID_VALUE                   -30
         printf("\nERROR: %s -- Invalid value in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_DEVICE_TYPE:             //#define CL_INVALID_DEVICE_TYPE             -31
         printf("\nERROR: %s -- Invalid device type in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_PLATFORM:                //#define CL_INVALID_PLATFORM                -32
         printf("\nERROR: %s -- Invalid platform in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_DEVICE:                  //#define CL_INVALID_DEVICE                  -33
         printf("\nERROR: %s -- Invalid device in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_CONTEXT:                 //#define CL_INVALID_CONTEXT                 -34
         printf("\nERROR: %s -- Invalid context in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_QUEUE_PROPERTIES:        //#define CL_INVALID_QUEUE_PROPERTIES        -35
         printf("\nERROR: %s -- Invalid queue properties in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_COMMAND_QUEUE:           //#define CL_INVALID_COMMAND_QUEUE           -36
         printf("\nERROR: %s -- Invalid command queue in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_HOST_PTR:                //#define CL_INVALID_HOST_PTR                -37
         printf("\nERROR: %s -- Invalid host pointer in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_MEM_OBJECT:              //#define CL_INVALID_MEM_OBJECT              -38
         printf("\nERROR: %s -- Invalid memory object in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: //#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR -39
         printf("\nERROR: %s -- Invalid image format descriptor in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_IMAGE_SIZE:              //#define CL_INVALID_IMAGE_SIZE              -40
         printf("\nERROR: %s -- Invalid image size in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_SAMPLER:                 //#define CL_INVALID_SAMPLER                 -41
         printf("\nERROR: %s -- Invalid sampler in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_BINARY:                  //#define CL_INVALID_BINARY                  -42
         printf("\nERROR: %s -- Invalid binary in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_BUILD_OPTIONS:           //#define CL_INVALID_BUILD_OPTIONS           -43
         printf("\nERROR: %s -- Invalid build options in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_PROGRAM:                 //#define CL_INVALID_PROGRAM                 -44
         printf("\nERROR: %s -- Invalid program in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_PROGRAM_EXECUTABLE:      //#define CL_INVALID_PROGRAM_EXECUTABLE      -45
         printf("\nERROR: %s -- Invalid program executable in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_KERNEL_NAME:             //#define CL_INVALID_KERNEL_NAME             -46
         printf("\nERROR: %s -- Invalid kernel name in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_KERNEL_DEFINITION:       //#define CL_INVALID_KERNEL_DEFINITION       -47
         printf("\nERROR: %s -- Invalid kernel definition in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_KERNEL:                  //#define CL_INVALID_KERNEL                  -48
         printf("\nERROR: %s -- Invalid kernel in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_ARG_INDEX:               //#define CL_INVALID_ARG_INDEX               -49
         printf("\nERROR: %s -- Invalid arg index in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_ARG_VALUE:               //#define CL_INVALID_ARG_VALUE               -50
         printf("\nERROR: %s -- Invalid arg value in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_ARG_SIZE:                //#define CL_INVALID_ARG_SIZE                -51
         printf("\nERROR: %s -- Invalid arg size in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_KERNEL_ARGS:             //#define CL_INVALID_KERNEL_ARGS             -52
         printf("\nERROR: %s -- Invalid kernel args in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_WORK_DIMENSION:          //#define CL_INVALID_WORK_DIMENSION          -53
         printf("\nERROR: %s -- Invalid work dimension in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_WORK_GROUP_SIZE:         //#define CL_INVALID_WORK_GROUP_SIZE         -54
         printf("\nERROR: %s -- Invalid work group size in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_WORK_ITEM_SIZE:          //#define CL_INVALID_WORK_ITEM_SIZE          -55
         printf("\nERROR: %s -- Invalid work item size in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_GLOBAL_OFFSET:           //#define CL_INVALID_GLOBAL_OFFSET           -56
         printf("\nERROR: %s -- Invalid global offset in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_EVENT_WAIT_LIST:         //#define CL_INVALID_EVENT_WAIT_LIST         -57
         printf("\nERROR: %s -- Invalid event wait list in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_EVENT:                   //#define CL_INVALID_EVENT                   -58
         printf("\nERROR: %s -- Invalid event in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_OPERATION:               //#define CL_INVALID_OPERATION               -59
         printf("\nERROR: %s -- Invalid operation in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_GL_OBJECT:               //#define CL_INVALID_GL_OBJECT               -60
         printf("\nERROR: %s -- Invalid GL object in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_BUFFER_SIZE:             //#define CL_INVALID_BUFFER_SIZE             -61
         printf("\nERROR: %s -- Invalid buffer size in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_MIP_LEVEL:               //#define CL_INVALID_MIP_LEVEL               -62
         printf("\nERROR: %s -- Invalid mip level in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;
      case CL_INVALID_GLOBAL_WORK_SIZE:        //#define CL_INVALID_GLOBAL_WORK_SIZE        -63
         printf("\nERROR: %s -- Invalid global work size in %s at line %d in file %s\n", routine, cl_routine, line, file);
         break;

      default:
        printf("\nERROR: %s -- %d in %s at line %d in file %s\n", routine, ierr, cl_routine, line, file);
        break;
   }
   void* callstack[128];
   int frames = backtrace(callstack, 128);
   if (frames > 2) {
#ifdef HAVE_ADDR2LINE
      char hex_address[21];
      const char command_string[80];
#endif
      char** strs = backtrace_symbols(callstack, frames);
      fprintf(stderr,"\n  =============== Backtrace ===============\n");
      for (int i = 1; i < frames-1; ++i) {
          fprintf(stderr,"   %s    \t", strs[i]);
#ifdef HAVE_ADDR2LINE
          sscanf(strs[i],"%*s [%[^]]s]",hex_address);
          //printf("DEBUG addr2line -e clamr -f -s -i -p %s\n",hex_address);
          sprintf(command_string,"addr2line -e clamr -f -s -i -p %s",hex_address);
          system(command_string);
          // on mac, need to install binutils using macports "port install binutils"
#endif
          fprintf(stderr,"\n");
      }
      fprintf(stderr,"  =============== Backtrace ===============\n\n");
      free(strs);
   }

   exit(-1);
}

