//
// Created by Salva Carrión on 11/10/2019.
//

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

Tensor* run_mpool1(Tensor* t_input, int dev){
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
    MPool2D(pd);

    return pd->O;
}

