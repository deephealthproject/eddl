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


    // Test pooling
    PoolDescriptor* pd;

    // Image
    float ptr_img[5*5] = {0, 1, 0, 4, 5,
                          2, 3, 2, 1, 3,
                          4, 4, 0, 4, 3,
                          2, 5, 2, 6, 4,
                          1, 0, 0, 5, 7};
    auto* t1 = new Tensor({1, 1, 5, 5}, ptr_img, device);
    t1->print();

    // [MaxPool] Sol. 1
    float ptr_mp_3x3_s1_padv[3*3] = {4,4,5,
                                     5,6,6,
                                     5,6,7};
    auto* t2_mp = new Tensor({1, 1, 3, 3}, ptr_mp_3x3_s1_padv, device);

    // [MaxPool] Sol. 2
    float ptr_mp_3x3_s1_pads[5*5] = {3,3,4,5,5,
                                     4,4,4,5,5,
                                     5,5,6,6,6,
                                     5,5,6,7,7,
                                     5,5,6,7,7};
    auto* t3_mp = new Tensor({1, 1, 5, 5}, ptr_mp_3x3_s1_pads, device);

    // [AvgPool] Sol. 1
    float ptr_ap_3x3_s2_padv[2*2] = {1.8, 2.4,
                                     2, 3.4};
    auto* t2_ap = new Tensor({1, 1, 2, 2}, ptr_ap_3x3_s2_padv, device);

    // [AvgPool] Sol. 2
    float ptr_ap_3x3_s2_pads[3*3] = {0.7, 1.2, 1.4,
                                     2.2, 3.0, 2.3,
                                     0.9, 2.0, 2.4};
    auto* t3_ap = new Tensor({1, 1, 3, 3}, ptr_ap_3x3_s2_pads, device);

    // [MaxPool] Test 1  ************
    cout << "*************************************" << endl;
    cout << "Result MaxPool(3x3_s1_padv):" << endl;
    pd = new PoolDescriptor({3, 3}, {1,1}, "none");
    pd->build(t1);
    pd->indX = new Tensor(pd->O->getShape(), device);
    pd->indY = new Tensor(pd->O->getShape(), device);

    // Forward
    MPool2D(pd);
    pd->O->print();

    cout << "Correct MaxPool(3x3_s1_padv):" << Tensor::equal2(t2_mp, pd->O, 10e-1f)  <<  endl;
    t2_mp->print();

    // [MaxPool] Test 2  ************
    cout << "*************************************" << endl;
    cout << "Result MaxPool(3x3_s1_pads):" << endl;
    pd = new PoolDescriptor({3, 3}, {1,1}, "same");
    pd->build(t1);
    pd->indX = new Tensor(pd->O->getShape(), device);
    pd->indY = new Tensor(pd->O->getShape(), device);

    // Forward
    MPool2D(pd);
    pd->O->print();

    cout << "Correct MaxPool(3x3_s1_pads):" << Tensor::equal2(t3_mp, pd->O, 10e-1f) << endl;
    t3_mp->print();

    // [AvgPool] Test 1  ************
    cout << "*************************************" << endl;
    cout << "Result AvgPool(3x3_s2_padv):" << endl;
    pd = new PoolDescriptor({3, 3}, {2,2}, "valid");
    pd->build(t1);

    // Forward
    AvgPool2D(pd);
    pd->O->print();

    cout << "Correct AvgPool(3x3_s2_padv):" << Tensor::equal2(t2_ap, pd->O, 10e-1f)  <<  endl;
    t2_ap->print();

    // [AvgPool] Test 2  ************
    cout << "*************************************" << endl;
    cout << "Result AvgPool(3x3_s2_padn):" << endl;
    pd = new PoolDescriptor({3, 3}, {2,2}, "same");
    pd->build(t1);

    // Forward
    AvgPool2D(pd);
    pd->O->print();

    cout << "Correct AvgPool(3x3_s2_pads):" << Tensor::equal2(t3_ap, pd->O, 10e-1f) << endl;
    t3_ap->print();

}
