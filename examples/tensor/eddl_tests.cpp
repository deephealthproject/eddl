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

    string path1 = "../examples/data/elephant.jpg";
    string output = "./output/";

    // Load image
    t1 = Tensor::load(path1);
    t1->unsqueeze_();
    t1->save(output + "output1.jpg");

    // Downscale
    t2 = Tensor::zeros({1, 3, 100, 100});
    Tensor::scale(t1, t2, {100, 100});
    t2->save(output + "output2.jpg");
    t1->set_select({":", ":", "100:200", "300:400"}, t2);  // "Paste" t2 in t1

    // Rotate
    t3 = t2->clone();
    Tensor::rotate(t2, t3, 60.0f, {0,0}, "original");
    t3->mult_(0.5f);
    t3->clampmax_(255.0f);
    t1->set_select({":", ":", "-150:-50", "-150:-50"}, t3);  // "Paste" t3 in t1


    // Save original
    t1->save(output + "output.jpg");
    int asdasd = 33;
}
