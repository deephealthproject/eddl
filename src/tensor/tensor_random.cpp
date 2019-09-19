#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

#include "tensor.h"
#include "../utils.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;


void Tensor::rand_uniform(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = uniform() * v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_uniform(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


void Tensor::rand_suniform(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = suniform() * v;

    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_suniform(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif


}


void Tensor::rand_binary(float v) {
    if (isCPU()) {
        for (int i = 0; i < size; ++i)
            if (uniform() < v) ptr[i] = 1.0;
            else ptr[i] = 0.0;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_binary(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


void Tensor::rand_gaussian(float m, float s) {
    if (isCPU()) {
        int r=rand();
        for (int i = 0; i < size; ++i) ptr[i] = gauss(r, m, s);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_gaussian(this,m,s);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}
