#include "tensor.h"
#include "../random.h"

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


void Tensor::rand_signed_uniform(float v) {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = signed_uniform() * v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_signed_uniform(this,v);
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


void Tensor::rand_normal(float m, float s, bool fast_math) {
    if (isCPU()) {
        if(fast_math){
            for (int i = 0; i < size; ++i) ptr[i] = fast_randn(m, s, rand());
        }else{
            for (int i = 0; i < size; ++i) ptr[i] = slow_randn(m, s);
        }

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
