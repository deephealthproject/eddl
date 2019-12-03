//
// Created by Salva Carrión on 11/10/2019.
//

#include <ctime>
#include <iostream>
#include <stdio.h>

#include "aux_tests.h"

#include "../../src/tensor/tensor.h"
#include "../../src/tensor/nn/tensor_nn.h"
#include "../../src/layers/core/layer_core.h"
#include "../../src/layers/normalization/layer_normalization.h"
#include "../../src/descriptors/descriptors.h"
#include "../../src/descriptors/descriptors.h"

#include "../../src/hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../src/hardware/gpu/gpu_tensor.h"
#include "../../src/hardware/gpu/gpu_hw.h"
#include "../../src/hardware/gpu/nn/gpu_nn.h"
#endif



bool check_tensors(Tensor* A, Tensor* B, float epsilon){
    // Clone input tensors
    A = A->clone();
    B = B->clone();

    // Copy to CPU (equal only supported in CPU)
    A->toCPU();
    B->toCPU();

    return Tensor::equal(A, B, epsilon);
}

TestResult run_mpool(Tensor* t_input, int dev, int runs){
    // Clone input tensor
    t_input = t_input->clone();

    // Move to device
    if (dev == DEV_GPU){
        t_input->toGPU();
    }

    // Instantiate PoolDescription + Perform MaxPooling
    auto *pd = new PoolDescriptor(vector<int>{2,2}, vector<int>{2,2}, "none");
    pd->build(t_input);
    pd->indX = new Tensor(pd->O->getShape(), dev);
    pd->indY = new Tensor(pd->O->getShape(), dev);

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        MPool2D(pd);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result;
    result.time = elapsed_secs;
    result.tensor = pd->O;
    return result;
}



TestResult run_conv2d(Tensor* t_input, Tensor* t_kernel, int dev, int runs){
    // Clone input tensor
    t_input = t_input->clone();
    t_kernel = t_kernel->clone();

    // Move to device
    if (dev == DEV_GPU){
        t_input->toGPU();
        t_kernel->toGPU();
    }

    // Instantiate PoolDescription + Perform MaxPooling
    auto *cd = new ConvolDescriptor(1, {3, 3}, {1, 1}, "none");
    cd->build(t_input);
    cd->K = t_kernel;
    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        Conv2D(cd);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = cd->O;
    return result;
}

TestResult run_dense(Tensor* t_input, Tensor* t_weights, int dev, int runs){
    // Clone input tensor
    t_input = t_input->clone();
    t_weights = t_weights->clone();
    Tensor *t_output = new Tensor(vector<int>{t_input->shape[0], t_weights->shape[1]}, dev);

    // Move to device
    if (dev == DEV_GPU){
        t_input->toGPU();
        t_weights->toGPU();
        t_output->toGPU();
    }

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        Tensor::mult2D(t_input, 0, t_weights, 0, t_output, 0);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = t_output;
    return result;
}


TestResult run_activation(Tensor* t_input, string act, int dev, int runs){
    // Clone input tensor
    t_input = t_input->clone();
    Tensor *t_output = new Tensor(t_input->getShape(), dev);

    // Move to device
    if (dev == DEV_GPU){
        t_input->toGPU();
        t_output->toGPU();
    }

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        if (act == "relu")
            ReLu(t_input, t_output);
        else if (act == "softmax") {
            Softmax(t_input, t_output);
        }
        else if (act == "sigmoid") {
            Sigmoid(t_input, t_output);
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = t_output;
    return result;
}



TestResult run_batchnorm(Tensor* t_input, int dev, int runs){
    // Clone input tensor
    t_input = t_input->clone();

    // Move to device
    if (dev == DEV_GPU){
        t_input->toGPU();
    }

    LTensor* l_t = new LTensor(t_input->getShape(), t_input->ptr, t_input->device);
    LBatchNorm* l_bn = new LBatchNorm(l_t, 0.9f, 0.001f, true, "", dev);

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        l_bn->forward();
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = l_bn->output;
    return result;
}


TestResult run_upsampling(Tensor* t_input, vector<int> size, int dev, int runs){
    // Clone input tensor
    t_input = t_input->clone();
    Tensor *t_output = new Tensor(t_input->getShape(), dev);

    // Move to device
    if (dev == DEV_GPU){
        t_input->toGPU();
        t_output->toGPU();
    }

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        repeat_nn(t_input, t_output, size);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = t_output;
    return result;
}


TestResult run_tensor_op(Tensor* t_input, string op, int dev, int runs){
    // Clone input tensor
    Tensor* t_output = t_input->clone();

    // Move to device
    if (dev == DEV_GPU){
        t_output->toGPU();
    }

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        // Math operations
        if(op=="abs"){ t_output->abs_(); }
        else if(op=="acos"){ t_output->acos_(); }
        else if(op=="add"){ t_output->add_(2.0f); }
        else if(op=="asin"){ t_output->asin_(); }
        else if(op=="atan"){ t_output->atan_(); }
        else if(op=="ceil"){ t_output->ceil_(); }
        else if(op=="clamp"){ t_output->clamp_(-0.5f, 0.5f); }
        else if(op=="cos"){ t_output->cos_(); }
        else if(op=="cosh"){ t_output->cosh_(); }
        else if(op=="exp"){ t_output->exp_(); }
        else if(op=="inv"){ t_output->acos_(); }
        else if(op=="floor"){ t_output->floor_(); }
        else if(op=="log"){ t_output->log_(); }
        else if(op=="log2"){ t_output->log2_(); }
        else if(op=="log10"){ t_output->log10_(); }
        else if(op=="logn"){ t_output->logn_(10.0f); }
        else if(op=="mod"){ t_output->mod_(5.0f); }
        else if(op=="mult"){ t_output->mult_(5.0f); }
        else if(op=="normalize"){ t_output->normalize_(0.0f, 1.0f); }
        else if(op=="pow"){ t_output->pow_(2.0f); }
        else if(op=="powb"){ t_output->powb_(2.0f); }
        else if(op=="reciprocal"){ t_output->reciprocal_(); }
        else if(op=="remainder"){ t_output->remainder_(5.0f); }
        else if(op=="round"){ t_output->round_(); }
        else if(op=="rsqrt"){ t_output->rsqrt_(); }
        else if(op=="sigmoid"){ t_output->sigmoid_(); }
        else if(op=="sign"){ t_output->sign_(); }
        else if(op=="sin"){ t_output->sin_(); }
        else if(op=="sinh"){ t_output->sinh_(); }
        else if(op=="sqr"){ t_output->sqr_(); }
        else if(op=="sqrt"){ t_output->sqrt_(); }
        else if(op=="tan"){ t_output->tan_(); }
        else if(op=="tanh"){ t_output->tanh_(); }
        else if(op=="trunc"){ t_output->trunc_(); }
        else if(op=="max"){ t_output->max(); }
        else if(op=="min"){ t_output->min(); }

        else{
            std::cout << "Unknown operator" << std::endl;
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = t_output;
    return result;
}

TestResult run_tensor_da(Tensor* t_input, Tensor* t_output, string op, int dev, int runs){
    // Clone input tensor
    t_input = t_input->clone();
    t_output = t_output->clone();

    // Move to device
    if (dev == DEV_GPU){
        t_input->toGPU();
        t_output->toGPU();
    }

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        if(op=="shift"){ Tensor::shift(t_input, t_output, {-1, -1}); }
        else if(op=="rotate"){ Tensor::rotate(t_input, t_output, 30.0f); }
        else if(op=="flip_v"){ Tensor::flip(t_input, t_output, 0); }
        else if(op=="flip_h"){ Tensor::flip(t_input, t_output,  1);}
        else if(op=="scale"){ Tensor::scale(t_input, t_output, {5, 5});}
        else if(op=="crop"){ Tensor::crop(t_input, t_output, {1,1}, {3, 3}); }
        else if(op=="crop_scale"){ Tensor::crop_scale(t_input, t_output, {1,1}, {3, 3}); }
        else if(op=="cutout"){ Tensor::cutout(t_input, t_output, {1, 1}, {3, 3});}

        else if(op=="shift_random"){ Tensor::shift_random(t_input, t_output, {-0.5f, +0.5f}, {-0.5f, +0.5f}); }
        else if(op=="rotate_random"){ Tensor::rotate_random(t_input, t_output, {-90.0f, +90.0f}); }
        else if(op=="flip_v_random"){ Tensor::flip_random(t_input, t_output, 0); }
        else if(op=="flip_h_random"){ Tensor::flip_random(t_input, t_output,  1);}
        else if(op=="scale_random"){ Tensor::scale_random(t_input, t_output, {2.0f, 2.0f});}
        else if(op=="crop_random"){ Tensor::crop_random(t_input, t_output);}
        else if(op=="crop_scale_random"){ Tensor::crop_scale_random(t_input, t_output, {0.5f, 1.0f});}
        else if(op=="cutout_random"){ Tensor::cutout_random(t_input, t_output, {0.1f, 0.5f}, {0.1f, 0.5f});}

        else{
            std::cout << "Unknown operator" << std::endl;
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = t_output;
    return result;
}


TestResult run_tensor_create(string op, int dev, int runs){
    Tensor* t_input;

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        // Math operations
        if(op=="zeros"){
            t_input = Tensor::zeros({10}, dev);
        }
        else if(op=="ones"){
            t_input = Tensor::ones({10}, dev);
        }
        else if(op=="full"){
            t_input = Tensor::full({10}, 5.0f, dev);
        }
        else if(op=="arange"){
            t_input = Tensor::arange(1.0f, 10.0f, 1.0f, dev);
        }
        else if(op=="range"){
            t_input = Tensor::range(1.0f, 10.0f, 1.0f, dev);
        }
        else if(op=="linspace"){
            t_input = Tensor::linspace(3.0f, 10.0f, 5, dev);
        }
        else if(op=="logspace"){
            t_input = Tensor::logspace(0.1f, 1.0f, 5, 10.0f, dev);
        }
        else if(op=="eye"){
            t_input = Tensor::eye(3, dev);
        }
        else if(op=="randn"){
            t_input = Tensor::randn({1000, 1000}, dev);
        }
        else{
            std::cout << "Unknown operator" << std::endl;
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = t_input;
    return result;
}

TestResult run_tensor_select(Tensor* t_input, Tensor* t_output, string op, int* oi_addresses, int dev, int runs){
    // Move to device
    if (dev == DEV_GPU){
        t_input->toGPU();
        t_output->toGPU();
    }

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        // Math operations
        if(op=="select") {
            Tensor::select(t_input, t_output, oi_addresses);
        }else if(op=="select_back") {
            Tensor::select_back(t_input, t_output, oi_addresses);
        } else {
            std::cout << "Unknown operator" << std::endl;
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    TestResult result{};
    result.time = elapsed_secs;
    result.tensor = t_output;
    return result;
}
