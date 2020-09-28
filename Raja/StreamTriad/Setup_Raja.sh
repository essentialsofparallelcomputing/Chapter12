# Raja

rm -rf Raja build

export INSTALL_DIR=`pwd`/Raja

cd Raja_build
rm -rf build
mkdir build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j 8 install
cd ../..

export Raja_DIR=${INSTALL_DIR}/share/raja/cmake

mkdir build && cd build && cmake .. && make && ./StreamTriad

make clean && make distclean

cd .. && rm -rf build Raja
