/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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
//    Tensor *t0 = Tensor::full({1, 3, 500, 500}, 255.0f, DEV_CPU);
    //t0->reshape_({1, 1, 100, 100});

    Tensor *t0 = Tensor::load("images/cow.jpg");
    t0->ToGPU();
    t0->info();
//    float* ptr = new float[3*4*2]{
//        255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f,
//        128.0f, 128.0f, 128.0f, 128.0f, 128.0f, 128.0f, 128.0f, 128.0f,
//        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
//    Tensor* t1 = new Tensor({1, 3, 4, 2}, ptr, DEV_CPU);
    Tensor *t1 = t0->clone();
    Tensor *t2 = t0->clone();


//     *************************************************
//     ***** Make color/light transformations **********
//     *************************************************
//    t1->add_(100);
//    t1->clampmax_(255);
//
//     *************************************************
//     ***** Make standard transformations *************
//     *************************************************

   // t1 = t0;
   // Tensor::shift(t1, t2, {10, 30});
   // t1 = t2->clone();

   // Tensor::rotate(t1, t2, 30.0f);
   // t1 = t2->clone();

   // float scale=1.25f;
   // Tensor::scale(t1, t2, {(int)(t2->shape[2]*scale), (int)(t2->shape[3]*scale)});
   // t1 = t2->clone();

   // Tensor::flip(t1, t2, 1);
   // t1 = t2->clone();

   // t2 = new Tensor({1, 3, 200, 400}, t0->device);
   // Tensor::crop(t1, t2, {0, 0}, {200, 400}); // Note: The final size depends on the size of t2
   // t1 = t2->clone();

   // Tensor::crop_scale(t1, t2, {20, 50}, {100, 150});
   // t1 = t2->clone();

   // Tensor::cutout(t1, t2, {80, 80}, {100, 200});
   // t1 = t2->clone();

   // t2->ToCPU();
   // t2->save("images/new_cow_single.jpg");
   // cout << "Image saved!" << endl;


//     *************************************************
//     ***** Make random transformations ***************
//     *************************************************
    for (int i = 1; i <= 10; i++) {
        t1 = t0->clone();
        t2 = t0->clone();
        //t1->info();

        Tensor::shift_random(t1, t2, {-0.3f, +0.3f}, {-0.3f, 0.3f});
        t1 = t2->clone();

        Tensor::rotate_random(t1, t2, {-30.0f, +30.0f});
        t1 = t2->clone();

        Tensor::scale_random(t1, t2, {0.5f, 2.0f});
        t1 = t2->clone();

        Tensor::flip_random(t1, t2, 1);
        t1 = t2->clone();

        t2 = new Tensor({1, 3, 100, 300}, t0->device);
        Tensor::crop_random(t1, t2);  //In pixels
        t1 = t2->clone();

        Tensor::crop_scale_random(t1, t2, {0.5f, 1.0f});
        t1 = t2->clone();

        Tensor::cutout_random(t1, t2, {0.1f, 0.3f}, {0.1, 0.3f});
        t1 = t2->clone();

        // Save result
        t2->ToCPU();
        t2->save("images/new_cow_" + to_string(i) + ".jpg");
        cout << "Image saved! #" << i << endl;
    }
}
