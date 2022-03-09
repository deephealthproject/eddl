/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

// Checking deletes, memory leaks
// CNN 3D Synchronous rnn, CPU, GPU
// In a separate terminal try
// top/htop and nvidia-smi (GPU)
// to check memory

int main(int argc, char **argv){

  int times_cpu=10;
  int times_gpu=100;

  //CPU
  for(int i=0;i<times_cpu;i++) {
    cout<<"======================="<<endl;
    cout<<"CPU "<<i<<endl;
    cout<<"======================="<<endl;

    layer in  = Input({3, 10, 64, 64});
    layer l=in;
     // Conv3D expects (B,C,dim1,dim2,dim3)
    l=MaxPool3D(ReLu(Conv3D(l,4,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=MaxPool3D(ReLu(Conv3D(l,8,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=MaxPool3D(ReLu(Conv3D(l,16,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=GlobalMaxPool3D(l);
    //l=Squeeze(l);
    l = Reshape(l, {-1});
    l = LSTM(l, 128);
    l = Dense(l, 100);
    l = ReLu(l);
    l = Dense(l, 2);
    layer out = ReLu(l);
    model net = Model({in},{out});

    build(net,
          adam(),
          {"mse"},
          {"mse"},
          CS_CPU()
          );

    // Input: 10 samples that are sequences of 10  3D RGB images of 64x64.
    Tensor* x_train = Tensor::randu({10, 10, 3, 10, 64, 64});

    // Target: A sequence of 7 samples of 2 values per image
    Tensor* y_train = Tensor::randu({10, 7, 2});


    fit(net, {x_train}, {y_train}, 10, 1);


    delete x_train;
    delete y_train;
    delete net;

  }

  //GPU
  for(int i=0;i<times_gpu;i++) {
    cout<<"======================="<<endl;
    cout<<"GPU "<<i<<endl;
    cout<<"======================="<<endl;

    layer in  = Input({3, 10, 64, 64});
    layer l=in;
     // Conv3D expects (B,C,dim1,dim2,dim3)
    l=MaxPool3D(ReLu(Conv3D(l,4,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=MaxPool3D(ReLu(Conv3D(l,8,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=MaxPool3D(ReLu(Conv3D(l,16,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=GlobalMaxPool3D(l);
    //l=Squeeze(l);
    l = Reshape(l, {-1});
    l = LSTM(l, 128);
    l = Dense(l, 100);
    l = ReLu(l);
    l = Dense(l, 2);
    layer out = ReLu(l);
    model net = Model({in},{out});

    build(net,
          adam(),
          {"mse"},
          {"mse"},
          CS_GPU({1})
          );

    // Input: 10 samples that are sequences of 10  3D RGB images of 64x64.
    Tensor* x_train = Tensor::randu({10, 10, 3, 10, 64, 64});

    // Target: A sequence of 7 samples of 2 values per image
    Tensor* y_train = Tensor::randu({10, 7, 2});


    fit(net, {x_train}, {y_train}, 10, 1);


    delete x_train;
    delete y_train;
    delete net;

  }

}
