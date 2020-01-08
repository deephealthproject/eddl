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

    Tensor* t1 = nullptr;
    Tensor* t2 = nullptr;
    Tensor* t3 = nullptr;
    Tensor* t4 = nullptr;

    string path1;
    string path2;
    string output = "./output/";

    path1 = "/Users/salvacarrion/Documents/Programming/C++/eddl/cmake-build-debug/output/output.jpg";
    path2 = "/Users/salvacarrion/Desktop/elephant2.jpg";

    t1 = Tensor::load(path1);
    t1->unsqueeze_();

    t2 = Tensor::load(path2);
    t2->unsqueeze_();


    t1->set_select({":", "-150:-50", "400:400"}, t2);

    t3 = t2->clone();
    Tensor::rotate(t2, t3, 60.0f);
    t3->mult_(0.5f);
    t3->clampmax_(255.0f);
    t1->set_select({":", ":", "100:200", "300:400"}, t3);

    t4 = new Tensor({1, 3, 50, 50});
    Tensor::scale(t2, t4, {50, 50});
    Tensor* t5 = t4->clone();
    Tensor::rotate(t5, t4, -200.0f);
    t4->mult_(1.75f);
    t4->clampmax_(255.0f);
    t1->set_select({":", ":", "300:350", "300:350"}, t4);


    t1->save(output + "output.jpg");
    int asdasd = 33;
}
