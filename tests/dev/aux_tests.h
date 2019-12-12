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

bool check_tensors(Tensor* A, Tensor* B, float epsilon=0.01f);

TestResult run_mpool(Tensor* t_input, int dev, int runs=1);
TestResult run_conv2d(Tensor* t_input, Tensor* t_kernel, int dev, int runs=1);
TestResult run_dense(Tensor* t_input, Tensor* t_weights, int dev, int runs);
TestResult run_activation(Tensor* t_input, string act, int dev, int runs=1);
TestResult run_batchnorm(Tensor* t_input, int dev, int runs=1);
TestResult run_upsampling(Tensor* t_input, vector<int> size, int dev, int runs=1);
TestResult run_tensor_op(Tensor* t_input, string op, int dev, int runs=1);
TestResult run_tensor_da(Tensor* t_input, Tensor* t_output, string op, int dev, int runs=1);
TestResult run_tensor_create(string op, int dev, int runs=1);
TestResult run_tensor_select(Tensor* t_input, Tensor* t_output, string op, int dev, int runs);

#endif //EDDL_AUX_TESTS_H
