//
// Created by Salva Carrión on 11/10/2019.
//

#include "aux_tests.h"
#include "../tensor/tensor.h"
#include "../tensor/nn/tensor_nn.h"
#include "../descriptors/descriptors.h"
#include "../descriptors/descriptors.h"


bool test_mpool(){
    // Set data
    float mpool_input[16] = {12.0, 20.0, 30.0, 0.0,
                             8.0, 12.0, 2.0, 0.0,
                             34.0, 70.0, 37.0, 4.0,
                             112.0, 100.0, 25.0, 12.0};
    float mpool_sol[4] = {20.0, 30.0,
                          112.0, 37.0};

    // Create tensors
    Tensor *t_cpu = new Tensor({1, 1, 4, 4}, mpool_input, DEV_CPU);
    Tensor *t_cpu_sol = new Tensor({1, 1, 2, 2}, mpool_sol, DEV_CPU);

    // [CPU] Instantiate PoolDescription + perform MaxPooling
    auto *pd_cpu = new PoolDescriptor(vector<int>{2,2}, vector<int>{2,2}, "none");
    pd_cpu->build(t_cpu);
    pd_cpu->indX = new Tensor(pd_cpu->O->getShape(), DEV_CPU);
    pd_cpu->indY = new Tensor(pd_cpu->O->getShape(), DEV_CPU);
    MPool2D(pd_cpu);

    return Tensor::equal(pd_cpu->O, t_cpu_sol);
}