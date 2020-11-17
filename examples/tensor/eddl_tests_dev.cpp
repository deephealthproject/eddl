/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
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

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
// [
// [12.00 inf nan]
// [-inf 0.00 inf]
// ]

    Tensor* r1 = t1->isfinite(); // returns new tensor

    r1->print(2);  // Temp.
// [
// [1.00 0.00 0.00]
// [0.00 1.00 0.00]
// ]


    cout << "Done!" << endl;

}
