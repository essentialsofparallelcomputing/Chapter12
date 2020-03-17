FROM ubuntu:18.04 AS builder
WORKDIR /project
RUN apt-get update && \
    apt-get install -y cmake git vim gcc g++ wget python3 xterm gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y cuda && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# We can trim down the install list to reduce the download time. Here is the list from Nvidia.
# cuda				Installs all CUDA Toolkit and Driver packages. Handles upgrading to the next version of the cuda package when it's released.
# cuda-10-2			Installs all CUDA Toolkit and Driver packages. Remains at version 10.2 until an additional version of CUDA is installed.
# cuda-toolkit-10-2		Installs all CUDA Toolkit packages required to develop CUDA applications. Does not include the driver.
# cuda-tools-10-2		Installs all CUDA command line and visual tools.
# cuda-runtime-10-2		Installs all CUDA Toolkit packages required to run CUDA applications, as well as the Driver packages.
# cuda-compiler-10-2		Installs all CUDA compiler packages.
# cuda-libraries-10-2		Installs all runtime CUDA Library packages.
# cuda-libraries-dev-10-2	Installs all development CUDA Library packages.
# cuda-drivers			Installs all Driver packages. Handles upgrading to the next version of the Driver packages when they're released.

RUN apt-get update && \
    apt-get install -y clinfo ocl-icd-libopencl1 ocl-icd-* opencl-headers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
RUN rm -f GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
RUN echo "deb https://apt.repos.intel.com/oneapi all main" >> /etc/apt/sources.list.d/oneAPI.list
RUN echo "deb [trusted=yes arch=amd64] https://repositories.intel.com/graphics/ubuntu bionic main" >> /etc/apt/sources.list.d/intel-graphics.list
RUN apt-get update && \
    apt-get install -y intel-basekit intel-hpckit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#apt-get install intel-opencl && \

RUN apt-get update && \
    apt-get install -y libnuma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add -
RUN echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' >> /etc/apt/sources.list.d/rocm.list
RUN apt-get update && \
    apt-get install -y rocm-opencl-dev rocm-dkms && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/triSYCL/triSYCL.git

RUN git clone https://github.com/kokkos/kokkos Kokkos_build && mkdir Kokkos_build/build
WORKDIR /project/Kokkos_build/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/Project/Kokkos -DKokkos_ENABLE_OPENMP=On && make install

WORKDIR /project
RUN git clone --recursive https://github.com/llnl/raja.git Raja_build && mkdir Raja_build/build
WORKDIR /project/Raja_build/build
RUN cmake ../ -DCMAKE_INSTALL_PREFIX=/Project/Raja && make install

SHELL ["/bin/bash", "-c"]
ENV Kokkos_DIR=/Project/Kokkos/lib/cmake/Kokkos
ENV Raja_DIR=/Project/Raja/share/raja/cmake
ENV PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV PATH=${PATH}:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64
ENV PATH=${PATH}:/opt/intel/inteloneapi/compiler/2021.1-beta04/linux/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/inteloneapi/compiler/2021.1-beta04/linux/compiler/lib/intel64_lin:/opt/intel/inteloneapi/compiler/2021.1-beta04/linux/lib

RUN groupadd chapter12 && useradd -m -s /bin/bash -g chapter12 chapter12

RUN usermod -a -G video chapter12

WORKDIR /home/chapter12
RUN chown -R chapter12:chapter12 /home/chapter12
USER chapter12

RUN git clone --recursive https://github.com/essentialsofparallelcomputing/Chapter12.git

ENTRYPOINT ["bash"]
