cmake_minimum_required (VERSION 3.1)
project (StreamTriad)

if (DEVICE_DETECT_DEBUG)
   add_definitions(-DDEVICE_DETECT_DEBUG=1)
endif (DEVICE_DETECT_DEBUG)

find_package(OpenCL REQUIRED)
set(HAVE_CL_DOUBLE ON CACHE BOOL "Have OpenCL Double")
set(NO_CL_DOUBLE OFF)
include_directories(${OpenCL_INCLUDE_DIRS})

# Adds build target of StreamTriad with source code files
add_executable(StreamTriad StreamTriad.c ezcl_lite.c ezcl_lite.h timer.c timer.h)
target_link_libraries(StreamTriad ${OpenCL_LIBRARIES})
add_dependencies(StreamTriad StreamTriad_kernel_source)

########### embed source target ###############
message(${CMAKE_CURRENT_BINARY_DIR})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/StreamTriad_kernel.inc
                  COMMAND ${CMAKE_SOURCE_DIR}/embed_source.pl ${CMAKE_SOURCE_DIR}/StreamTriad_kernel.cl > StreamTriad_kernel.inc
                  DEPENDS StreamTriad_kernel.cl ${CMAKE_SOURCE_DIR}/embed_source.pl)
add_custom_target(StreamTriad_kernel_source ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/StreamTriad_kernel.inc)

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake StreamTriad.dSYM ipo_out.optrpt)

SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "StreamTriad_kernel.inc")
