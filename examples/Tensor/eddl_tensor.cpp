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
//#include <omp.h>

#include "apis/eddlT.h"
#include "../../src/tensor/tensor_reduction.h"

using namespace std;
using namespace eddlT;

int main(int argc, char **argv) {
    vector<int> shape = {5, 5};

    int device = DEV_GPU;
//    cout << "CAncel " << omp_get_cancellation() << endl;

    Tensor *t1 = Tensor::ones({3, 4, 5});
    t1->print();
    t1->info();
    Tensor::swapaxis(t1, 0, 2);

    t1->reshape_({2, 2, -1});
    t1->print();
    t1->info();

    t1 = Tensor::flatten(t1);
    t1->print();
    t1->info();

    cout << Tensor::isSquared(t1) << endl;
    int asd =3;
//
//    Tensor* t1 = Tensor::range(1.0, 25.0f, 1, device);
//    t1->reshape_(shape);
//    t1->print();
//
//    Tensor* t2 = Tensor::range(1.0, 25.0f, 1, device);
//    t2->reshape_(shape);
//    t2->print();
//
//    Tensor* t3 = new Tensor(shape, device);
//
//    cout << "allclose" << endl;
//    cout << Tensor::allclose(t1, t2) << endl;
//
//    cout << "isclose" << endl;
//    Tensor::isclose(t1, t2, t3);
//    t3->print();
//
//    cout << "greater" << endl;
//    Tensor::greater(t1, t2, t3);
//    t3->print();
//
//    cout << "greater_equal" << endl;
//    Tensor::greater_equal(t1, t2, t3);
//    t3->print();
//
//    cout << "less" << endl;
//    Tensor::less(t1, t2, t3);
//    t3->print();
//
//    cout << "less_equal" << endl;
//    Tensor::less_equal(t1, t2, t3);
//    t3->print();
//
//    cout << "equal" << endl;
//    Tensor::equal(t1, t2, t3);
//    t3->print();
//
//    cout << "not_equal" << endl;
//    Tensor::not_equal(t1, t2, t3);
//    t3->print();
//
//
//    cout << "------------" << endl;
//    t1->fill_(1.0);
//    t2->fill_(0.0);
//
//    t1->round_();
//    t2->round_();
//
//    t1->print();
//    t2->print();
//
//    cout << "logical_and" << endl;
//    Tensor::logical_and(t1, t2, t3);
//    t3->print();
//
//    cout << "logical_not" << endl;
//    Tensor::logical_not(t1, t3);
//    t3->print();
//
//    cout << "logical_or" << endl;
//    Tensor::logical_or(t1, t2, t3);
//    t3->print();
//
//    cout << "logical_xor" << endl;
//    Tensor::logical_xor(t1, t2, t3);
//    t3->print();
//
//    int asdas=33;

//    string fname = "datasets/drive/numpy/x_train.npy";
//    t1 = Tensor::load<uint8_t>(fname);
//    t1->info();
//    cout << "Max: " << t1->max() << endl;
//    cout << "Min: " << t1->min() << endl;
//    t2 = t1->select({"0"});
//    t2->info();
////    t2->unsqueeze_();
//    t2 = Tensor::permute(t2, {0, 3, 1, 2});
//    t2->info();
//    t2->save("numpy_ds.jpg");
//    int asd = 3;

//    float ptr[12] = {1, 2, 3,  1, 2, 3,
//                     1, 2, 3,  1, 2, 3};
//    t1= new Tensor({1, 2, 2, 3}, ptr, DEV_CPU);
//
//    t1->print();
//    t2 = Tensor::permute(t1, {0, 3, 1, 2});
//    t2->print();
//    int asd = 33;
//
//    t1 = Tensor::range(1, 16);
//    t1->reshape_({4, 4});
//    t1->print();
//
//    t2 = t1->select({":", "1:3"});
//    t2->print();
//
//    t1 = Tensor::moveaxis(t1, 0, 1);
//    t1->reshape_({4, 4});
//    t1->print();
//    int aasd = 33;
//
//    string fname = "/Users/salvacarrion/Desktop/elephant.jpg";
//    t1 = Tensor::load(fname);
//    t2 = new Tensor(t1->shape);
//
//    t1->save("rotate1.jpg");
//    Tensor::rotate(t1, t2, 45, {0,0}, "original");
//    t2->save("rotate2.jpg");
//    int as33d = 33;
//
//  int dev=DEV_GPU;
//  vector<int> axis={0,2,3};
//
//
//  Tensor *A=new Tensor({32,64,224,224},dev);
//  Tensor *B=new Tensor({64},dev);
//
//
//  A->fill_(2.0);
//  int *map=get_reduction_map(A, axis);
//
//  reduce_mean(A,B,axis,map);
//  B->print();
//
//  int devc=DEV_CPU;
//  Tensor *Ac=new Tensor({32,64,224,224},devc);
//  Tensor *Bc=new Tensor({64},devc);
//
//  Ac->fill_(2.0);
//
//  reduce_mean(Ac,Bc,axis,map);
//  Bc->print();
//
//  B->toCPU();
//  if (!Tensor::equal(B,Bc,0.1)) {
//    fprintf(stderr,"Error not equal\n");
//  }

  //B->print();

  //reduce_mult(A,B,axis);

  //A->print();

  /*
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
    t0->toGPU();
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

   // t2->toCPU();
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
        t2->toCPU();
        t2->save("images/new_cow_" + to_string(i) + ".jpg");
        cout << "Image saved! #" << i << endl;
    }
    */
}
