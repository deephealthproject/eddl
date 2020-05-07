/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
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


#include "eddl/tensor/tensor_reduction.h"

using namespace std;


int main(int argc, char **argv) {
    Tensor* t1 = nullptr;
    Tensor* t2 = nullptr;
    Tensor* t3 = nullptr;

    vector<int> shape = {5, 5};

    // ***************************************************
    // Create tensor *************************************
    // ***************************************************
    t1 = Tensor::ones({5, 1});
    cout << "Tensor 1: ********************" << endl;
    t1->info();
    t1->print();
    cout << endl;

    t2 = Tensor::full({5, 1}, 5);
    cout << "Tensor 2: ********************" << endl;
    t2->info();
    t2->print();
    cout << endl;


    // ***************************************************
    // Test: Array contents 1 ****************************
    // ***************************************************
    cout << endl;
    cout << "Test: Array contents 1 ********************" << endl;

    t1->ptr[0] = 12; // isfinite
    t1->ptr[1] = INFINITY; // isinf
    t1->ptr[2] = NAN; // isnan
    t1->ptr[3] = -INFINITY; // isneginf
    t1->ptr[4] = +INFINITY; // isposinf
    cout << "New tensor 1: ********************" << endl;
    t1->print();
    cout << endl;

    cout << "isfinite" << endl;
    Tensor::isfinite(t1, t2); t2->print();
    cout << endl;

    cout << "isinf" << endl;
    Tensor::isinf(t1, t2); t2->print();
    cout << endl;

    cout << "isnan" << endl;
    Tensor::isnan(t1, t2); t2->print();
    cout << endl;

    cout << "isneginf" << endl;
    Tensor::isneginf(t1, t2); t2->print();
    cout << endl;

    cout << "isposinf" << endl;
    Tensor::isposinf(t1, t2); t2->print();
    cout << endl;

    // ***************************************************
    // Test: Array contents 2 ****************************
    // ***************************************************
    cout << endl;
    cout << "Test: Array contents 2 ********************" << endl;
    cout << endl;

    t1 = Tensor::ones({3, 3}); t1->print();
    cout << "All? " << Tensor::all(t1) << endl;
    cout << endl;

    t1 = Tensor::zeros({3, 3}); t1->print();
    cout << "Any? " << Tensor::any(t1) << endl;
    cout << endl;

    // ***************************************************
    // Test: Array contents 3 ****************************
    // ***************************************************
    cout << endl;
    cout << "Test: Array contents 3 ********************" << endl;
    cout << endl;

    cout << "New Tensor 1: ********************" << endl;
    t1 = Tensor::range(1.0, 25.0f, 1);
    t1->reshape_(shape); t1->print();
    cout << endl;

    cout << "New Tensor 2: ********************" << endl;
    t2 = Tensor::range(1.0, 25.0f, 1);
    t2->reshape_(shape); t2->print();
    cout << endl;

    cout << "allclose" << endl;
    cout << Tensor::allclose(t1, t2) << endl;
    cout << endl;

    cout << "isclose" << endl;
    t3 = new Tensor(shape);
    Tensor::isclose(t1, t2, t3);
    t3->print();
    cout << endl;

    cout << "greater" << endl;
    Tensor::greater(t1, t2, t3); t3->print();
    cout << endl;

    cout << "greater_equal" << endl;
    Tensor::greater_equal(t1, t2, t3); t3->print();
    cout << endl;

    cout << "less" << endl;
    Tensor::less(t1, t2, t3); t3->print();
    cout << endl;

    cout << "less_equal" << endl;
    Tensor::less_equal(t1, t2, t3); t3->print();
    cout << endl;

    cout << "equal" << endl;
    Tensor::equal(t1, t2, t3); t3->print();
    cout << endl;

    cout << "not_equal" << endl;
    Tensor::not_equal(t1, t2, t3); t3->print();
    cout << endl;

    // ***************************************************
    // Test: Array contents 4 ****************************
    // ***************************************************
    cout << endl;
    cout << "Test: Array contents 4 ********************" << endl;
    cout << endl;

    cout << "New Tensor 1: ********************" << endl;
    t1->fill_(1.0);  t1->print();
    cout << endl;

    cout << "New Tensor 2: ********************" << endl;
    t2->fill_(0.0);  t1->print();
    cout << endl;

    cout << "logical_and" << endl;
    Tensor::logical_and(t1, t2, t3); t3->print();

    cout << "logical_not" << endl;
    Tensor::logical_not(t1, t3); t3->print();
    cout << endl;

    cout << "logical_or" << endl;
    Tensor::logical_or(t1, t2, t3); t3->print();
    cout << endl;

    cout << "logical_xor" << endl;
    Tensor::logical_xor(t1, t2, t3); t3->print();
    cout << endl;


}
