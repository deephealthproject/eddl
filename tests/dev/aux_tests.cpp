//
// Created by Salva Carrión on 11/10/2019.
//

#include <ctime>

#include "aux_tests.h"

#include "../../src/tensor/tensor.h"
#include "../../src/tensor/nn/tensor_nn.h"
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
