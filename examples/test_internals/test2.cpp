/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

// Checking deletes, memory leaks
// RNN models, CPU and GPU
// In a separate terminal try
// top/htop and nvidia-smi (GPU)
// to check memory

int main(int argc, char **argv){

  // download CIFAR data
  // download_cifar10();

  // network
  int times=100;

  int ilength=30;
  int olength=30;
  int invs=687;
  int outvs=514;
  int embedding=64;

  //CPU
  for(int i=0;i<times;i++) {
    cout<<"======================="<<endl;
    cout<<"CPU "<<i<<endl;
    cout<<"======================="<<endl;

    // Encoder
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = RandomUniform(Embedding(l, invs, 1,embedding,true),-0.05,0.05); // mask_zeros=true
    layer enc = LSTM(lE,128,true);  // mask_zeros=true
    layer cps=GetStates(enc);

    // Decoder
    layer ldin=Input({outvs});
    layer ld = ReduceArgMax(ldin,{0});
    ld = RandomUniform(Embedding(ld, outvs, 1,embedding),-0.05,0.05);

    // input from embedding and
    // state from encoder
    l = LSTM({ld,cps},128);
    layer out = Softmax(Dense(l, outvs));

    setDecoder(ldin);

    model net = Model({in}, {out});

    optimizer opt=adam(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          CS_CPU()
    );

      // Load dataset
    Tensor *x_train=Tensor::zeros({10,ilength,1}); //batch x timesteps x input_dim
    Tensor *y_train=Tensor::zeros({10,olength,outvs}); //batch x timesteps x ouput_dim

    // to force unrolling
    fit(net, {x_train}, {y_train}, 10, 1);


    delete x_train;
    delete y_train;
    delete net;
  }

  //GPU
  for(int i=0;i<times;i++) {
    cout<<"======================="<<endl;
    cout<<"GPU "<<i<<endl;
    cout<<"======================="<<endl;
    // Encoder
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = RandomUniform(Embedding(l, invs, 1,embedding,true),-0.05,0.05); // mask_zeros=true
    layer enc = LSTM(lE,128,true);  // mask_zeros=true
    layer cps=GetStates(enc);

    // Decoder
    layer ldin=Input({outvs});
    layer ld = ReduceArgMax(ldin,{0});
    ld = RandomUniform(Embedding(ld, outvs, 1,embedding),-0.05,0.05);

    // input from embedding and
    // state from encoder
    l = LSTM({ld,cps},128);
    layer out = Softmax(Dense(l, outvs));

    setDecoder(ldin);

    model net = Model({in}, {out});

    optimizer opt=adam(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
    );

      // Load dataset
    Tensor *x_train=Tensor::zeros({10,ilength,1}); //batch x timesteps x input_dim
    Tensor *y_train=Tensor::zeros({10,olength,outvs}); //batch x timesteps x ouput_dim


    // to force unrolling
    fit(net, {x_train}, {y_train}, 10, 1);


    delete x_train;
    delete y_train;
    delete net;
  }

}
