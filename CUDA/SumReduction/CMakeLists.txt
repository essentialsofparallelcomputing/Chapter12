cmake_minimum_required (VERSION 2.8)
project (SumReduction)

find_package(CUDA REQUIRED)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CUDA_STANDARD 11)

# sets CMAKE_{C,CXX}_FLAGS from CUDA compile flags. Includes DEBUG and RELEASE
set (CUDA_PROPAGATE_HOST_FLAGS ON) # default is on
set (CUDA_SEPARABLE_COMPILATION ON) # default is off

if (CMAKE_VERSION VERSION_GREATER "3.9.0")
   cuda_select_nvcc_arch_flags(ARCH_FLAGS)
endif()

set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 ${ARCH_FLAGS})

# Adds build target of SumReduction with source code files
cuda_add_executable(SumReduction SumReduction.cu timer.c timer.h)

if (APPLE)
    set_property(TARGET SumReduction PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake SumReduction.dSYM ipo_out.optrpt
                  detect_cuda_compute_capabilities.cpp)


# Adds a make clean_cuda_depends target -- invoke with "make clean_cuda_depends"
CUDA_BUILD_CLEAN_TARGET()
