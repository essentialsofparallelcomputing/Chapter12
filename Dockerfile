FROM ubuntu:18.04 AS builder
WORKDIR /project
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
    apt-get -qq install -y cmake git vim gcc g++ gfortran software-properties-common \
            wget gnupg python3 xterm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Installing latest GCC compiler (version 8) for latest compatible with CUDA
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get -qq update && \
    apt-get -qq install -y gcc-8 g++-8 gfortran-8 \
                           gcc-9 g++-9 gfortran-9 \
                           gcc-10 g++-10 gfortran-10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# We can use the update-alternatives to switch between compiler versions
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 90\
                        --slave /usr/bin/g++ g++ /usr/bin/g++-8\
                        --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-8\
                        --slave /usr/bin/gcov gcov /usr/bin/gcov-8

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 80\
                        --slave /usr/bin/g++ g++ /usr/bin/g++-9\
                        --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-9\
                        --slave /usr/bin/gcov gcov /usr/bin/gcov-9

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 70\
                        --slave /usr/bin/g++ g++ /usr/bin/g++-10\
                        --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-10\
                        --slave /usr/bin/gcov gcov /usr/bin/gcov-10

RUN chmod u+s /usr/bin/update-alternatives

# Intel graphics software for computation
RUN wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
RUN rm -f GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
RUN echo "deb https://apt.repos.intel.com/oneapi all main" >> /etc/apt/sources.list.d/oneAPI.list
RUN echo "deb [trusted=yes arch=amd64] https://repositories.intel.com/graphics/ubuntu bionic main" >> /etc/apt/sources.list.d/intel-graphics.list

RUN apt-get -qq update && \
    apt-get -qq install -y \
            intel-basekit-getting-started \
            intel-oneapi-advisor \
            intel-oneapi-ccl \
            intel-oneapi-common-licensing \
            intel-oneapi-common-vars \
            intel-oneapi-dev-utilities \
            intel-oneapi-dpcpp-compiler \
            intel-oneapi-dpcpp-ct \
            intel-oneapi-dpcpp-debugger \
            intel-oneapi-ipp \
            intel-oneapi-ipp-devel \
            intel-oneapi-libdpstd-devel \
            intel-oneapi-mkl \
            intel-oneapi-mkl-devel \
            intel-oneapi-openmp \
            intel-oneapi-tbb \
            intel-oneapi-tbb-devel \
            intel-oneapi-vtune \
            intel-oneapi-icc \
            intel-oneapi-icc-eclipse-cfg \
            intel-hpckit-getting-started \
            intel-oneapi-ifort \
            intel-oneapi-inspector \
            intel-oneapi-itac \
            intel-opencl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Generic OpenCL Loader
RUN apt-get -qq update && \
    apt-get -qq install -y clinfo ocl-icd-libopencl1 ocl-icd-* opencl-headers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Nvidia GPU software for computation
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
RUN apt-get -qq update && \
    apt-get -qq install -y cuda-toolkit-10-2 cuda-tools-10-2 cuda-compiler-10-2 \
        cuda-libraries-10-2 cuda-libraries-dev-10-2 libnvidia-compute-450 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ROCm software installation
RUN apt-get -qq update && \
    apt-get -qq install -y libnuma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add -
RUN echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' >> /etc/apt/sources.list.d/rocm.list
RUN apt-get -qq update && \
    apt-get -qq install -y rocm-opencl-dev rocm-dkms && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Vendor OpenCL
RUN apt-get -qq update && \
    apt-get -qq install -y mesa-opencl-icd && \
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

RUN groupadd chapter12 && useradd -m -s /bin/bash -g chapter12 chapter12

RUN usermod -a -G video chapter12

WORKDIR /home/chapter12
RUN chown -R chapter12:chapter12 /home/chapter12
USER chapter12
ENV Kokkos_DIR=/Project/Kokkos/lib/cmake/Kokkos
ENV Raja_DIR=/Project/Raja/share/raja/cmake
ENV PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV PATH=${PATH}:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64
ENV PATH=${PATH}:/opt/intel/inteloneapi/compiler/2021.1-beta04/linux/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/inteloneapi/compiler/2021.1-beta04/linux/compiler/lib/intel64_lin:/opt/intel/inteloneapi/compiler/2021.1-beta04/linux/lib

RUN git clone --recursive https://github.com/essentialsofparallelcomputing/Chapter12.git

WORKDIR /home/chapter12/Chapter12
#RUN make

ENTRYPOINT ["bash"]
