#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, fOBree, rand */

#include "../tensor.h"


//int EDDL_DEV=DEV_CPU;
int EDDL_DEV=DEV_GPU;

int main(int argc, char **argv)
{

  Tensor *A=new Tensor(10,10,10);
  Tensor *B=new Tensor(10,10,20);

  if (Tensor::eqsize(A,B)) printf("iguales\n");
  else printf("diferentes\n");

}
