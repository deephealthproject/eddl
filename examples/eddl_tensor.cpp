
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

#include "apis/eddl.h"

using namespace eddl;
using namespace std;

int main(int argc, char **argv) {
    // TEST: ones
    cout << "\n" << "ones: =============" << endl;
    auto t = Tensor::ones({10});
    t->info();
    t->print();

    // TEST: zeros
    cout << "\n" << "zeros: =============" << endl;
    t = Tensor::zeros({10});
    t->info();
    t->print();

    // TEST: arange
    cout << "\n" << "arange: =============" << endl;
    t = Tensor::arange(0.0, 1.0, 0.5);
    t->info();
    t->print();

    // TEST: Linear space
    cout << "\n" << "linspace: =============" << endl;
    t = Tensor::linspace(3.0, 10.0, 5);
    t->info();
    t->print();

    // TEST: Linear space
    cout << "\n" << "eye: =============" << endl;
    t = Tensor::eye(3);
    t->info();
    t->print();


}

