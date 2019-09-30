//
// Created by Salva Carrión on 30/09/2019.
//

#ifndef EDDL_CPU_HW_H
#define EDDL_CPU_HW_H

#include "../../tensor/tensor.h"


void cpu_transpose(Tensor *A, Tensor *B);
void cpu_copy(Tensor *A, Tensor *B);
void cpu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);
void cpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);

#endif //EDDL_CPU_HW_H
