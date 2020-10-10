# Chapter 12 GPU languages: getting down to basics
This is from Chapter 12 of Parallel and High Performance Computing, Robey and Zamora,
Manning Publications, available at http://manning.com

The book may be obtained at
   http://www.manning.com/?a_aid=ParallelComputingRobey

Copyright 2019-2020 Robert Robey, Yuliana Zamora, and Manning Publications
Emails: brobey@earthlink.net, yzamora215@gmail.com

See License.txt for licensing information.

CUDA StreamTriad (Book: listing 12.1 - 12.6)
   Build with cmake and make
      cd CUDA/StreamTriad
      mkdir build && cd build
      cmake .. && make
   Run with
      ./StreamTriad

CUDA SumReduction (Book: listing 12.7 - 12.10)
   Build with cmake
      cd CUDA/SumReduction
      mkdir build && cd build
      cmake .. && make
   Run with
      ./SumReduction

CUDA/SumReductionRevealed
   Build with make
      make
   Run with
      ./SumReductionRevealed

HIP StreamTriad (Book: listing 12.11 - 12.13)
   Build with make or cmake
      cd HIP/StreamTriad
      ln -s Makefile.perl Makefile
      make
        or
      mkdir build && cd build
      cmake .. && make
   Run with
      ./StreamTriad

OpenCL StreamTriad (Book: listing 12.14 - 12.19)
   Build with make or cmake
      cd OpenCL/StreamTriad
      mkdir build && cd build
      cmake .. && make
   Run with
      ./StreamTriad

OpenCL SumReduction (Book: listing 12.20)
   Build with cmake
      mkdir build && cd build
      cmake .. && make
   Run with
      ./SumReduction

SYCL (Book: listing 12.21 - 12.22)
   Build with make
      cd DPCPP/StreamTriad
      make
   Run with
      ./StreamTriad
      
Kokkos (Book: listing 12.23 - 12.24)
   Build with cmake
      mkdir build && cd build
      cmake .. && make
   Run with
      ./StreamTriad

Raja (Book: listing 12.25 - 12.27)
   Integrated build script that builds Raja and code
   Build with cmake
      mkdir build && cd build
      cmake .. && make
   Run with
      ./StreamTriad
