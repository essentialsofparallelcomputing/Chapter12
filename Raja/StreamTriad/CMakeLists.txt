cmake_minimum_required (VERSION 3.8)
project (StreamTriad)

include(blt/SetupBLT.cmake)

find_package(Raja REQUIRED)

blt_add_executable(NAME StreamTriad SOURCES StreamTTriad.cc timer.c)

#add_executable(StreamTriad StreamTriad.cc timer.c)
#target_link_libraries(StreamTriad Raja::raja)
