FROM ubuntu AS builder
WORKDIR /project
RUN apt-get update && \
    apt-get install -y bash cmake git vim gcc g++ wget python3 xterm gnupg
#RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
#RUN apt-key add /var/cuda-repo-ubuntu1804_10.2.89-1_amd64/7fa2af90.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af90.pub
#RUN apt-get update
#RUN apt-get install cuda
#RUN apt-get install clinfo ocl-icd-libopencl1 ocl-icd
#RUN cd /tmp; wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB; apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB; cd -
#RUN apt-get update
#RUN apt-get install intel-basekit -y
#RUN apt-get install intel-hpckit -y
RUN useradd -m chapter12
RUN echo "chapter12\n chapter12\n" > passwd chapter12
RUN echo "PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1:${PATH}" >> /home/chapter12/.bash_profile
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64 ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /home/chapter12/.bash_profile
RUN echo "source /opt/intel/inteloneapi/setvars.sh" >> /home/chapter12/.bash_profile
#RUN echo "deb [trusted=yes arch=amd64] https://repositories.intel.com/graphics/ubuntu bionic main" >> /etc/apt/sources.list.d/intel-graphics.list
RUN apt-get update
#RUN apt-get install -y intel-opencl

USER chapter12

#RUN cd; git clone https://github.com/triSYCL/triSYCL.git

RUN cd; git clone --recursive https://github.com/essentialsofparallelcomputing/Chapter12.git

RUN cd; git clone https://github.com/kokkos/kokkos Kokkos_build; cd Kokkos_build; mkdir build && cd build; cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos -DKokkos_ENABLE_OPENMP=On; make install; cd -
#RUN "export Kokkos_DIR=${HOME}/Kokkos/lib/cmake/Kokkos" >> /home/chapter12/.bash_profile

RUN cd; git clone --recursive https://github.com/llnl/raja.git Raja_build; cd Raja_build && mkdir build && cd build && cmake ../ -DCMAKE_INSTALL_PREFIX=${HOME}/Raja && make install && cd -

#RUN "export Raja_DIR=${HOME}/Raja/share/raja/cmake" >> /home/chapter12/.bash_profile

RUN bash

ENTRYPOINT ["bash"]
