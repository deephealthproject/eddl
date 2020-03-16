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
#include "../src/apis/eddl.h"

using namespace std;
using namespace eddl;

int main(int argc, char **argv) {
    cout << "Tests for development. Ignore." << endl;

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
    PoolDescriptor *pd;

/*    int o;
    vector<int> p;
    o = pd->compute_output("none", 8,1,3);
    p = pd->compute_padding(o, 8,1,3, "none");
    cout << "Output: " << o << endl;
    cout << "Padding: " << (p[0]+p[1]) << " (" << p[0] << ", " << p[1] << ")"<< endl;

    o = pd->compute_output("same", 8,1,3);
    p = pd->compute_padding(o,8,1,3, "same");
    cout << "Output: " << o << endl;
    cout << "Padding: " << (p[0]+p[1]) << " (" << p[0] << ", " << p[1] << ")"<< endl;

    o = pd->compute_output("none", 8,3,3);
    p = pd->compute_padding(o,8,3,3, "none");
    cout << "Output: " << o << endl;
    cout << "Padding: " << (p[0]+p[1]) << " (" << p[0] << ", " << p[1] << ")"<< endl;

    o = pd->compute_output("same", 8,3,3);
    p = pd->compute_padding(o,8,3,3, "same");
    cout << "Output: " << o << endl;
    cout << "Padding: " << (p[0]+p[1]) << " (" << p[0] << ", " << p[1] << ")"<< endl;*/

    layer in = Input({3, 16, 16});
    layer t = Transpose(in);

    bool use_gpu = true;

    // Image
    float *ptr_img = new float[5*5]{1, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7};
    auto* t1 = new Tensor({1, 1, 5, 5}, ptr_img);
    if(use_gpu) t1->toGPU();

    // [MaxPool] Sol. 1
    float *ptr_mp_3x3_s1_padv = new float[3*3]{4,4,5,
                                               5,6,6,
                                               5,6,7};
    auto* t2_mp = new Tensor({1, 1, 3, 3}, ptr_mp_3x3_s1_padv);
    if(use_gpu) t2_mp->toGPU();

    // [MaxPool-back] Sol. 1
    float *ptr_mp_3x3_s1_padv_back = new float[5*5]{0, 0, 0, 2, 4,
                                                    0, 2, 0, 0, 0,
                                                    2, 0, 0, 0, 0,
                                                    0, 6, 0, 5, 0,
                                                    0, 0, 0, 0, 4};
    auto* t2_mp_back = new Tensor({1, 1, 5, 5}, ptr_mp_3x3_s1_padv_back);
    if(use_gpu) t2_mp_back->toGPU();

    // [MaxPool] Sol. 2
    float *ptr_mp_3x3_s1_pads = new float[5*5]{3,3,4,5,5,
                                               4,4,4,5,5,
                                               5,5,6,6,6,
                                               5,5,6,7,7,
                                               5,5,6,7,7};
    auto* t3_mp = new Tensor({1, 1, 5, 5}, ptr_mp_3x3_s1_pads);
    if(use_gpu) t3_mp->toGPU();

    // [AvgPool] Sol. 1
    float *ptr_ap_3x3_s2_padv = new float[2*2]{1.8, 2.4,
                                               2, 3.4};
    auto* t2_ap = new Tensor({1, 1, 2, 2}, ptr_ap_3x3_s2_padv);
    if(use_gpu) t2_ap->toGPU();

    // [AvgPool] Sol. 2
    float *ptr_ap_3x3_s2_pads = new float[3*3]{0.7, 1.2, 1.4,
                                               2.2, 3.0, 2.3,
                                               0.9, 2.0, 2.4};
    auto* t3_ap = new Tensor({1, 1, 3, 3}, ptr_ap_3x3_s2_pads);
    if(use_gpu) t3_ap->toGPU();

    cout << "*************************************" << endl;
    cout << "Input image:" << endl;
    t1->print();

    // [MaxPool] Test 1  ************
    cout << endl;
    cout << "*************************************" << endl;
    cout << "Result MaxPool(3x3_s1_padv):" << endl;
    pd = new PoolDescriptor({3, 3}, {1,1}, "same");
    pd->build(t1);
    pd->indX = new Tensor(pd->O->getShape());  if(use_gpu) pd->indX->toGPU();
    pd->indY = new Tensor(pd->O->getShape());  if(use_gpu) pd->indY->toGPU();
    pd->ID = Tensor::zeros(pd->I->getShape()); if(use_gpu) pd->ID->toGPU();
    pd->D = Tensor::ones(pd->O->getShape());  if(use_gpu) pd->D->toGPU();

    // Forward
    MPool2D(pd);
    pd->O->print();

/*    t2_mp->toCPU();
    pd->O->toCPU();
    cout << "Correct MaxPool(3x3_s1_padv):" << Tensor::equal2(t2_mp, pd->O, 10e-1f)  <<  endl;
    t2_mp->print();
    */
    // [MaxPool Back] Test 1  ************
    cout << endl;
    cout << "*************************************" << endl;
    cout << "Result MaxPool-Back(3x3_s1_padv):" << endl;
    MPool2D_back(pd);
    pd->ID->print();

    t2_mp_back->toCPU();
    pd->ID->toCPU();
    cout << "Correct MaxPool-Back(3x3_s1_pads):" << Tensor::equal2(t2_mp_back, pd->ID, 10e-1f) << endl;
    t2_mp_back->print();


    // [MaxPool] Test 2  ************
    cout << endl;
    cout << "*************************************" << endl;
    cout << "Result MaxPool(3x3_s1_pads):" << endl;
    pd = new PoolDescriptor({3, 3}, {1,1}, "same");
    pd->build(t1);
    pd->indX = new Tensor(pd->O->getShape());  if(use_gpu) pd->indX->toGPU();
    pd->indY = new Tensor(pd->O->getShape());  if(use_gpu) pd->indY->toGPU();
    pd->ID = Tensor::zeros(pd->I->getShape()); if(use_gpu) pd->ID->toGPU();
    pd->D = Tensor::ones(pd->O->getShape());  if(use_gpu) pd->D->toGPU();

    // Forward
    MPool2D(pd);
    pd->O->print();

    t3_mp->toCPU();
    pd->O->toCPU();
    cout << "Correct MaxPool(3x3_s1_pads):" << Tensor::equal2(t3_mp, pd->O, 10e-1f) << endl;
    t3_mp->print();

    // [AvgPool] Test 1  ************
    cout << endl;
    cout << "*************************************" << endl;
    cout << "Result AvgPool(3x3_s2_padv):" << endl;
    pd = new PoolDescriptor({3, 3}, {2,2}, "valid");
    pd->build(t1);

    // Forward
    AvgPool2D(pd);
    pd->O->print();

    t2_ap->toCPU();
    pd->O->toCPU();
    cout << "Correct AvgPool(3x3_s2_padv):" << Tensor::equal2(t2_ap, pd->O, 10e-1f)  <<  endl;
    t2_ap->print();

    // [AvgPool] Test 2  ************
    cout << endl;
    cout << "*************************************" << endl;
    cout << "Result AvgPool(3x3_s2_padn):" << endl;
    pd = new PoolDescriptor({3, 3}, {2,2}, "same");
    pd->build(t1);

    // Forward
    AvgPool2D(pd);
    pd->O->print();

    t3_ap->toCPU();
    pd->O->toCPU();
    cout << "Correct AvgPool(3x3_s2_pads):" << Tensor::equal2(t3_ap, pd->O, 10e-1f) << endl;
    t3_ap->print();

}
