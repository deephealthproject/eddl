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


bool test_mpool(int dev){
    // Set data
    float mpool_input[16] = {12.0, 20.0, 30.0, 0.0,
                             8.0, 12.0, 2.0, 0.0,
                             34.0, 70.0, 37.0, 4.0,
                             112.0, 100.0, 25.0, 12.0};
    float mpool_sol[4] = {20.0, 30.0,
                          112.0, 37.0};

    // Create tensors
    Tensor *t = new Tensor({1, 1, 4, 4}, mpool_input, dev);
    Tensor *t_sol = new Tensor({1, 1, 2, 2}, mpool_sol, dev);

    // Instantiate PoolDescription + Perform MaxPooling
    auto *pd = new PoolDescriptor(vector<int>{2,2}, vector<int>{2,2}, "none");
    pd->build(t);
    pd->indX = new Tensor(pd->O->getShape(), dev);
    pd->indY = new Tensor(pd->O->getShape(), dev);
    MPool2D(pd);

    return Tensor::equal(pd->O, t_sol);
}