/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <ctime>
#include <limits>
#include <cmath>

//#include <omp.h>

#include "../src/initializers/initializer.h"
#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_reduction.h"
#include "../src/tensor/nn/tensor_nn.h"
#include "../src/regularizers/regularizer.h"
#include "../src/tensor/nn/tensor_nn.h"
#include "../src/hardware/cpu/nn/cpu_nn.h"

using namespace std;

int main(int argc, char **argv) {
    cout << "Tests for development. Ignore." << endl;

    int device = DEV_CPU;

    // Overload operations
//    Tensor t1 = *Tensor::full({5,5}, 1.0f, device);
//    Tensor t2 = *Tensor::full({5,5}, 2.0f, device);
//    Tensor t3 = ((t1 / t1) + (t2 * t2)) + 10;
//
//    t1.print();
//    t2.print();
//    t3.print();
//
//    // Mixed
//    t3 = ((t1 / t1) + (t2 * t2));
//
//    // Tensor op Scalar
//    t3 = t3 + 10;
//    t3 = t3 - 10;
//    t3 = t3 * 10;
//    t3 = t3 / 10;
//
//    // Scalar op Tensor
//    t3 = 10 + t3;
//    t3 = 10 - t3;
//    t3 = 10 * t3;
//    t3 = 10 / t3;
//
//    // Tensor op= Tensor
//    t3 += t2;
//    t3 -= t2;
//    t3 *= t2;
//    t3 /= t2;
//
//    // Tensor op= Scalar
//    t3 += 5;
//    t3 -= 5;
//    t3 *= 5;
//    t3 /= 5;
//

    // Test concat
//    Tensor* t5 = Tensor::range(1, 0+3*2*2, 1.0f, device); t5->reshape_({3, 2, 2}); t5->print();
//    Tensor* t6 = Tensor::range(11, 10+3*2*2, 1.0f, device); t6->reshape_({3, 2, 2}); t6->print();
//    Tensor* t7 = Tensor::range(101, 100+3*2*2, 1.0f, device); t7->reshape_({3, 2, 2}); t7->print();
//
//    // Concat
//    Tensor* t8 = Tensor::concat({t5, t6, t7}, 2);
//    t8->print();
//
//     Tensor::concat_back(t8, {t5, t6, t7}, 2);
//     t5->print();
//     t6->print();
//     t7->print();

    // Test average pooling
    float ptr[4*4] = {31, 15, 28, 184, 0, 100, 70, 38, 12, 12, 7, 2, 12, 12 ,45, 6};
    auto* t1 = new Tensor({1, 1, 4, 4}, ptr, device);

    auto* pd = new PoolDescriptor({2, 2}, {2,2}, "none");
    pd->build(t1);
    pd->indX = new Tensor(pd->O->getShape(), device);
    pd->indY = new Tensor(pd->O->getShape(), device);

    // Forward
    AvgPool2D(pd);

    // Print
    pd->O->print();
    int asda = 33;
}
