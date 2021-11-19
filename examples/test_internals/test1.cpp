/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

// Checking deletes, memory leaks
// CNN models, CPU and GPU
// In a separate terminal try
// top/htop and nvidia-smi (GPU)
// to check memory

layer BG(layer l) {
  return GaussianNoise(BatchNormalization(l),0.3);
  //return l;
}

layer ResBlock(layer l, int filters,int nconv,int half) {
  layer in=l;

  if (half)
      l=ReLu(BG(Conv(l,filters,{3,3},{2,2})));
  else
      l=ReLu(BG(Conv(l,filters,{3,3},{1,1})));


  for(int i=0;i<nconv-1;i++)
    l=ReLu(BG(Conv(l,filters,{3,3},{1,1})));

  if (half)
    return Add(BG(Conv(in,filters,{1,1},{2,2})),l);
  else
    return Add(l,in);
}

int main(int argc, char **argv){

  int times_cpu=100;
  int times_gpu=100;

  //CPU
  for(int i=0;i<times_cpu;i++) {
    cout<<"======================="<<endl;
    cout<<"CPU "<<i<<endl;
    cout<<"======================="<<endl;


    layer in=Input({3,32,32});
    layer l=in;

    l=ReLu(BG(Conv(l,64,{3,3},{1,1})));

    l=ResBlock(l, 64,2,1);
    l=ResBlock(l, 64,2,0);

    l=ResBlock(l, 128,2,1);
    l=ResBlock(l, 128,2,0);

    l=ResBlock(l, 256,2,1);
    l=ResBlock(l, 256,2,0);

    l=ResBlock(l, 256,2,1);
    l=ResBlock(l, 256,2,0);

    l=Reshape(l,{-1});
    l=ReLu(BG(Dense(l,512)));

    layer out= Softmax(Dense(l,10));

    model net=Model({in},{out});

    compserv cs = CS_CPU();

    build(net,
      sgd(0.001, 0.9), // Optimizer
      {"softmax_cross_entropy"}, // Losses
      {"categorical_accuracy"}, // Metrics
      cs);

    // Load dataset
    Tensor *x_train=Tensor::zeros({10,3,32,32});
    Tensor *y_train=Tensor::zeros({10,10});

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


    layer in=Input({3,32,32});
    layer l=in;

    l=ReLu(BG(Conv(l,64,{3,3},{1,1})));

    l=ResBlock(l, 64,2,1);
    l=ResBlock(l, 64,2,0);

    l=ResBlock(l, 128,2,1);
    l=ResBlock(l, 128,2,0);

    l=ResBlock(l, 256,2,1);
    l=ResBlock(l, 256,2,0);

    l=ResBlock(l, 256,2,1);
    l=ResBlock(l, 256,2,0);

    l=Reshape(l,{-1});
    l=ReLu(BG(Dense(l,512)));

    layer out= Softmax(Dense(l,10));

    model net=Model({in},{out});

    compserv cs = CS_GPU({1});

    build(net,
      sgd(0.001, 0.9), // Optimizer
      {"softmax_cross_entropy"}, // Losses
      {"categorical_accuracy"}, // Metrics
      cs);

    // Load dataset
    Tensor *x_train=Tensor::zeros({10,3,32,32});
    Tensor *y_train=Tensor::zeros({10,10});

    fit(net, {x_train}, {y_train}, 10, 1);

    delete x_train;
    delete y_train;

    delete net;
  }



}
