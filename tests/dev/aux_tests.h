//
// Created by Salva Carrión on 11/10/2019.
//

#ifndef EDDL_AUX_TESTS_H
#define EDDL_AUX_TESTS_H

#include "../../src/tensor/tensor.h"

bool check_tensors(Tensor* t_res, Tensor* t_sol);

Tensor* run_mpool1(Tensor* t_input, int dev);

#endif //EDDL_AUX_TESTS_H
