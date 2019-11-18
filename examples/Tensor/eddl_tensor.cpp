/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <limits>

#include "apis/eddlT.h"

using namespace std;
using namespace eddlT;

int main(int argc, char **argv) {
//    int dev = DEV_CPU;
//
//    tensor A=create({10,10});
//    fill_(A,0.0);
//
//    tensor T=randn({10,10},dev);
//
//    print(T);
//
//    add_(A,T);
//    print(A);
//
//    normalize_(T,0,1);
//
//    print(T);
//
//    tensor U=randn({10,3},dev);
//
//    print(U);
//
//    tensor V=mult2D(T,U);
//
//    info(V);
//
//    print(V);


    // Open image
    cout<<"AQUI\n";
    Tensor* t1 = Tensor::load("images/cow.jpg", "jpg");
//    float* ptr = new float[3*4*2]{
//        255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f,
//        128.0f, 128.0f, 128.0f, 128.0f, 128.0f, 128.0f, 128.0f, 128.0f,
//        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
//    Tensor* t1 = new Tensor({1, 3, 4, 2}, ptr, DEV_CPU);
    Tensor* t2 = t1->clone();
    t2->info();
    cout<<"AQUI\n";

//     *************************************************
//     ***** Make color/light transformations **********
//     *************************************************
//    t1->add_(100);
//    t1->clampmax_(255);
//
//     *************************************************
//     ***** Make standard transformations *************
//     *************************************************

//    Tensor::shift(t1, t2, {50, 20});
//    t1 = t2->clone();

    Tensor::rotate(t1, t2, 30.0f, {0, 0});
    t1 = t2->clone();
//
//    float scale=1.25f;
//    Tensor::scale(t1, t2, {(int)(t2->shape[2]*scale), (int)(t2->shape[3]*scale)});
//    t1 = t2->clone();
//
//    Tensor::flip(t1, t2, 1);
//    t1 = t2->clone();
//
////    Tensor::crop(t1, t2, {10, 0}, {60, 80}); // Note: The final size depends on the size of t2
////    t1 = t2->clone();
//
//    Tensor::crop_scale(t1, t2, {0, 0}, {400, 400});
//    t1 = t2->clone();
//
//    Tensor::cutout(t1, t2, {50, 50}, {100, 150});
//    t1 = t2->clone();


    // *************************************************
    // ***** Make random transformations ***************
    // *************************************************
//    Tensor::shift_random(t1, t2, {-0.3f, +0.3f}, {-0.3f, +0.3f});
//    t1 = t2->clone();
//
////    Tensor::rotate_random(t1, t2, {-0.3f, +0.3f}, {0, 1}});
////    t1 = t2->clone();
//
//    Tensor::scale_random(t1, t2, {1.0f, 2.0f});
//    t1 = t2->clone();
//
//    Tensor::flip_random(t1, t2, 1);
//    t1 = t2->clone();
//
////    Tensor::crop_random(t1, t2);  //In pixels
////    t1 = t2->clone();
//
//    Tensor::crop_scale_random(t1, t2, {0.5f, 1.0f});
//    t1 = t2->clone();
//
//    Tensor::cutout_random(t1, t2, {0.1f, 0.5f}, {0.1, 0.5f});
//    t1 = t2->clone();

    // Save result
    t2->save("images/new_cow.png", "png");
    cout << "Image saved!" << endl;
}
