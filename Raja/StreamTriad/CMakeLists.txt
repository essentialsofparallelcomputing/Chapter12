cmake_minimum_required (VERSION 3.0)
project (StreamTriad)

find_package(Raja REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(StreamTriad StreamTriad.cc)
target_link_libraries(StreamTriad PUBLIC RAJA)
set_target_properties(StreamTriad PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
set_target_properties(StreamTriad PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles Raja Raja_build/build
    Makefile cmake_install.cmake StreamTriad.dSYM ipo_out.optrpt)
