/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <ctime>
#include <limits>
#include <cmath>

//#include <omp.h>

#include "eddl/initializers/initializer.h"
#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/regularizers/regularizer.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/apis/eddl.h"

using namespace std;
using namespace eddl;

int main(int argc, char **argv) {
    cout << "Tests for development. Ignore." << endl;
    Tensor* t1 = new Tensor({1, 2, 3}, {3}, DEV_CPU);
    Tensor* t2 = new Tensor({1, 2, 4}, {3}, DEV_CPU);
    Tensor* t3 = Tensor::stack({t1, t2}, 1);
    t3->print(2);
    int asd = 3;



//    Tensor* t1 = Tensor::load("mnist_trY.bin");
//    Tensor* t2 = t1->select({"2:5"});
//    t2->print(2);
//
//
//    Tensor* t3 = Tensor::load_partial("mnist_trY.bin", 2, 5);
//
//    cout << Tensor::equivalent(t2, t3) << endl;
//    t3->print(2);


    cout << "Done!" << endl;

}
