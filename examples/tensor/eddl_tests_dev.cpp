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

#include "apis/eddlT.h"
#include "../../src/tensor/tensor_reduction.h"

using namespace std;
using namespace eddlT;

int main(int argc, char **argv) {

    cout << "Tests for development. Ignore." << endl;

    Tensor t1 = *Tensor::full({5,5}, 1.0f);
    Tensor t2 = *Tensor::full({5,5}, 2.0f);
    Tensor t3;

    t1.print();
    t2.print();

    t3 = ((t1 / t1) + (t2 * t2));

    // Tensor op Scalar
    t3 = t3 + 10; t3.print();
    t3 = t3 - 10; t3.print();
    t3 = t3 * 10; t3.print();
    t3 = t3 / 10; t3.print();

    // Scalar op Tensor
    t3 = 10 + t3; t3.print();
    t3 = 10 - t3; t3.print();
    t3 = 10 * t3; t3.print();
    t3 = 10 / t3; t3.print();

    // Tensor op= Tensor
    t3 += t2; t3.print();
    t3 -= t2; t3.print();
    t3 *= t2; t3.print();
    t3 /= t2; t3.print();

    // Tensor op= Scalar
    t3 += 5; t3.print();
    t3 -= 5; t3.print();
    t3 *= 5; t3.print();
    t3 /= 5; t3.print();

    int asda = 33;
}
