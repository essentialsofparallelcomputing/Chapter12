cmake_minimum_required (VERSION 2.8.3)
project (stream_triad)

set (CMAKE_CXX_STANDARD 11)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP REQUIRED)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(MY_HIPCC_OPTIONS )
set(MY_HCC_OPTIONS )
set(MY_NVCC_OPTIONS )

# Adds build target of stream_triad with source code files
HIP_ADD_EXECUTABLE(StreamTriad StreamTriad.cc timer.c timer.h)
target_include_directories(StreamTriad PRIVATE ${HIP_PATH}/include)
target_link_directories(StreamTriad PRIVATE ${HIP_PATH}/lib)
target_link_libraries(StreamTriad hip_hcc)

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles *.o
    Makefile cmake_install.cmake StreamTriad.dSYM ipo_out.optrpt)
