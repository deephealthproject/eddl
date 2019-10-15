//
// Created by Salva Carrión on 11/10/2019.
//

#ifndef EDDL_AUX_TESTS_H
#define EDDL_AUX_TESTS_H

#include "../../src/tensor/tensor.h"

struct TestResult{
    double time;
    Tensor* tensor;
};

bool check_tensors(Tensor* t_res, Tensor* t_sol);

TestResult run_mpool1(Tensor* t_input, int dev, int runs=1);

#endif //EDDL_AUX_TESTS_H
