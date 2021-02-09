/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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

//   Tensor* t1 = Tensor::load("/home/salvacarrion/Documents/Programming/C++/eddl/nan_tensor_lout_input.bin");
    Tensor *t1 = new Tensor({-114.67 ,-153.77 ,-122.57 ,-113.86 ,-141.96 ,-119.93 ,-116.40 ,-135.25 ,-105.31 ,-117.21}, {1, 10}, DEV_CPU);
   t1->print(2);

   Tensor* t2 = Tensor::zeros_like(t1);
//   t2 = t1->exp();
   tensorNN::FullSoftmax(t1, t2, 1);

   t2->print(2);

   delete t1;
   delete t2;

    cout << "Done!" << endl;

}
