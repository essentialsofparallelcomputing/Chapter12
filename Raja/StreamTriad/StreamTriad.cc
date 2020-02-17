#include "RAJA/RAJA.hpp"

using namespace std;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA daxpy example...\n";

// Define vector length
  const int N = 1000000;

// Allocate and initialize vector data.
  double* a0 = new double[N];
  double* aref = new double[N];

  double* ta = new double[N];
  double* tb = new double[N];
  
  double c = 3.14159;
  
  for (int i = 0; i < N; i++) {
    a0[i] = 1.0;
    tb[i] = 2.0;
  }

//
// Declare and set pointers to array data. 
// We reset them for each daxpy version so that 
// they all look the same.
//

  double* a = ta;
  double* b = tb;


//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of daxpy...\n";
   
  std::memcpy( a, a0, N * sizeof(double) );  

  for (int i = 0; i < N; ++i) {
    a[i] += b[i] * c;
  }

  std::memcpy( aref, a, N* sizeof(double) ); 

//----------------------------------------------------------------------------//

//
// In the following, we show a RAJA version
// of the daxpy operation and how it can
// be run differently by choosing different
// RAJA execution policies. 
//
// Note that the only thing that changes in 
// these versions is the execution policy.
// To implement these cases using the 
// programming model choices directly, would
// require unique changes for each.
//
  
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential daxpy...\n";
   
  std::memcpy( a, a0, N * sizeof(double) );  

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
    a[i] += b[i] * c;
  });

