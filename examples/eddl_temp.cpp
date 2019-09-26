
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
    int N = 10000;

    clock_t begin;
    clock_t end;
    double elapsed_secs;

    auto t1 = new Tensor({1000}, DEV_CPU);
    t1->set(5.0);

    // Test #ref
    begin = clock();
    for(int i=0; i<N; i++){
        t1->sqr_();
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "TEST #REF: " << std::to_string(elapsed_secs) << " seconds" << std::endl;


    // Test #1
    begin = clock();
    for(int i=0; i<N; i++){
        t1->pow_(2.0f);
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "TEST #1: " << std::to_string(elapsed_secs) << " seconds" << std::endl;


}

