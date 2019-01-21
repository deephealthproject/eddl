#ifndef _TENSOR_
#define _TENSOR_

#define DEV_CPU 0
#define DEV_GPU 1
#define DEV_FPGA 2

#include "cpu/Eigen/Dense"


using namespace Eigen;
using namespace std;

class Tensor {

  public:
  int device;
  int dim;
  int tam;
  int size[5]; // Up to 5D Tensors
  Tensor **ptr;

  // CPU
  RowVectorXd ptr1;
  MatrixXd ptr2;
  ////

  // GPU
  float *g_ptr;
  //

  Tensor();
  ~Tensor();


  Tensor(int a);
  Tensor(int a,int b);
  Tensor(int a,int b,int c);
  Tensor(int a,int b,int c,int d);
  Tensor(int a,int b,int c,int d,int e);


  static int eqsize(Tensor *A, Tensor *B);

};


#endif
