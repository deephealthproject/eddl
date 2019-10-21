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
#include "../../src/descriptors/descriptors.h"
#include "../../src/descriptors/descriptors.h"

#include "../../src/hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../src/hardware/gpu/gpu_tensor.h"
#include "../../src/hardware/gpu/gpu_hw.h"
#include "../../src/hardware/gpu/nn/gpu_nn.h"
#endif



bool check_tensors(Tensor* t_res, Tensor* t_sol){
    // Clone input tensors
    t_res = t_res->clone();
    t_sol = t_sol->clone();

    // Copy to CPU (equal only supported in CPU)
    t_res->ToCPU();
    t_sol->ToCPU();

    return Tensor::equal(t_res, t_sol);
}

TestResult run_mpool(Tensor* t_input, int dev, int runs){
    // Clone input tensor
    t_input = t_input->clone();

    // Move to device
    if (dev == DEV_GPU){
        t_input->ToGPU();
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
        t_input->ToGPU();
        t_kernel->ToGPU();
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
        t_input->ToGPU();
        t_weights->ToGPU();
        t_output->ToGPU();
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
        t_input->ToGPU();
        t_output->ToGPU();
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
        t_input->ToGPU();
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
        t_input->ToGPU();
        t_output->ToGPU();
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
    t_input = t_input->clone();

    // Move to device
    if (dev == DEV_GPU){
        t_input->ToGPU();
    }

    clock_t begin = clock();
    for(int i=0; i<runs; i++){
        if(op=="abs"){ t_input->abs_(); }
        else if(op=="acos"){ t_input->acos_(); }
        else if(op=="add"){ t_input->add_(2.0f); }
        else if(op=="asin"){ t_input->asin_(); }
        else if(op=="atan"){ t_input->atan_(); }
        else if(op=="ceil"){ t_input->ceil_(); }
        else if(op=="clamp"){ t_input->clamp_(-0.5f, 0.5f); }
        else if(op=="cos"){ t_input->cos_(); }
        else if(op=="cosh"){ t_input->cosh_(); }
        else if(op=="exp"){ t_input->exp_(); }
        else if(op=="inv"){ t_input->acos_(); }
        else if(op=="floor"){ t_input->floor_(); }
        else if(op=="log"){ t_input->log_(); }
        else if(op=="log2"){ t_input->log2_(); }
        else if(op=="log10"){ t_input->log10_(); }
        else if(op=="logn"){ t_input->logn_(10.0f); }
        else if(op=="mod"){ t_input->mod_(5.0f); }
        else if(op=="mult"){ t_input->mult_(5.0f); }
        else if(op=="normalize"){ t_input->normalize_(0.0f, 1.0f); }
        else if(op=="pow"){ t_input->pow_(2.0f); }
        else if(op=="reciprocal"){ t_input->reciprocal_(); }
        else if(op=="remainder"){ t_input->remainder_(5.0f); }
        else if(op=="round"){ t_input->round_(); }
        else if(op=="rsqrt"){ t_input->rsqrt_(); }
        else if(op=="sigmoid"){ t_input->sigmoid_(); }
        else if(op=="sign"){ t_input->sign_(); }
        else if(op=="sin"){ t_input->sin_(); }
        else if(op=="sinh"){ t_input->sinh_(); }
        else if(op=="sqr"){ t_input->sqr_(); }
        else if(op=="sqrt"){ t_input->sqrt_(); }
        else if(op=="tan"){ t_input->tan_(); }
        else if(op=="tanh"){ t_input->tanh_(); }
        else if(op=="trunc"){ t_input->trunc_(); }
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

