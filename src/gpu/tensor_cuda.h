#ifndef _tensor_cuda_
#define _tensor_cuda_

#include <cuda.h>


float* create_tensor(int size);
void delete_tensor(float* p);

#endif
