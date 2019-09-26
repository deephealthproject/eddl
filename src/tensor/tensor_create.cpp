#include "tensor.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;

//TODO: Move. Aux func I don't like it
Tensor* raw_range(float min, float step, int size, int dev){
    auto t = new Tensor(vector<int>{size}, nullptr, dev);
    float v=min;
    for(int i=0; i<size; i++){
        t->ptr[i] = v;
        v+=step;
    }
    return t;
}

// ************************************************
// Creation ops ***********************************
// ************************************************

Tensor* Tensor::zeros(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, nullptr, dev);
    t->set(0.0f);
    return t;
}

Tensor* Tensor::ones(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, nullptr, dev);
    t->set(1.0f);
    return t;
}

Tensor* Tensor::full(const vector<int> &shape, float value, int dev){
    auto t = new Tensor(shape, nullptr, dev);
    t->set(value);
    return t;
}

Tensor* Tensor::arange(float min, float max, float step, int dev){
    // Returns a 1-D tensor of size floor(end - start)/ + 1 with values from start to end with step step.
    // Step is the gap between two values in the tensor.
    int size = ceilf((max-min)/step);
    return raw_range(min, step, size, dev);
}

Tensor* Tensor::range(float min, float max, float step, int dev){
    int size = floorf((max-min)/step) + 1;
    return raw_range(min, step, size, dev);
}

Tensor* Tensor::linspace(float start, float end, int steps, int dev){
    float step = (end-start)/((float)steps-1);
    return Tensor::range(start, end, step, dev);
}

Tensor* Tensor::logspace(float start, float end, int steps, float base, int dev){
    float step = (end-start)/((float)steps-1);
    auto t = Tensor::range(start, end, step, dev);
    for(int i=0; i<steps; i++){
        t->ptr[i] = std::pow(base, t->ptr[i]);
    }
    return t;
}

Tensor* Tensor::eye(int size, int dev){
    auto t = new Tensor(vector<int>{size, size}, nullptr, dev);
    //t->set(0.0f);
    for(int i=0; i<size; i++){
        t->ptr[i*size+i] = 1.0f;
    }
    return t;
}

Tensor* Tensor::randn(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, nullptr, dev);
    t->rand_normal(0.0f, 1.0f, false);
    return t;
}
