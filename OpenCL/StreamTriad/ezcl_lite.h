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
#ifndef _EZCL_H
#define _EZCL_H

#ifdef __APPLE_CC__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

#define EZCL_SUCCESS         0
#define EZCL_NODEVICE        1

#define COMPUTE_DEVICE_NVIDIA   1
#define COMPUTE_DEVICE_AMD      2
#define COMPUTE_DEVICE_INTEL    3

#ifdef __cplusplus
extern "C"
{
#endif

// Profiling interfaces -- file and line number are inserted at these interfaces to give users
// better error messages
/* init and end routines */
#define ezcl_devtype_init(  device_type, command_queue, context) \
      ( ezcl_devtype_init_p(device_type, command_queue, context, __FILE__, __LINE__) )

/* kernel and program routines */
#define ezcl_create_program_wsource(  context, defines, source) \
      ( ezcl_create_program_wsource_p(context, defines, source, __FILE__, __LINE__) )

/* Error reporting */
#define ezcl_print_error(ierr, cl_routine) \
      ( ezcl_print_error_p(ierr, cl_routine, cl_routine, __FILE__, __LINE__) )

/* init and finish routine */
cl_int ezcl_devtype_init_p(cl_device_type device_type, cl_command_queue *command_queue, cl_context *context,
                           const char *file, const int line);

/* kernel and program routines */
cl_program ezcl_create_program_wsource_p(cl_context context, const char *defines, const char *source, const char *file, const int line);

/* Error reporting */
void ezcl_print_error_p(const int ierr, const char *routine, const char *cl_routine, const char *file, const int line);

   
#ifdef __cplusplus
}
#endif

#endif  /* _EZCL_H */
