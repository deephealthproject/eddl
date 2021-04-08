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


#include "eddl/tensor/tensor_reduction.h"

using namespace std;


int main(int argc, char **argv) {
    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_();  // 4D tensor needed

    // Pad
    Tensor* t2 = t1->pad({10, 20, 40, 80}, 255.0f);  // (top, right, bottom, left)
    t2->save("lena_pad22.jpg");
}
