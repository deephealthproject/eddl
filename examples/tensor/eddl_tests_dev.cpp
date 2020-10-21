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

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_();  // 4D tensor needed

    // Cutout
    Tensor* t2 = t1->cutout({50, 250}, {250, 400});
    t2->save("lena_cutout.jpg");

    // Other ways
    Tensor::cutout(t1, t2, {50, 250}, {250, 400});  // Static

    cout << "Done!" << endl;

}
