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
// CNN decoder models, CPU, GPU
// In a separate terminal try
// top/htop and nvidia-smi (GPU)
// to check memory

layer ResBlock(layer l, int filters,int nconv,int half) {
  layer in=l;

  if (half)
      l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{2,2})));
  else
      l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{1,1})));


  for(int i=0;i<nconv-1;i++)
    l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{1,1})));

  if (half)
    return Add(BatchNormalization(Conv(in,filters,{1,1},{2,2})),l);
  else
    return Add(l,in);
}

int main(int argc, char **argv){

  int times_cpu=5;
  int times_gpu=100;

  int ilength=30;
  int olength=30;
  int invs=687;
  int outvs=514;
  int embedding=64;

  //CPU
  for(int i=0;i<times_cpu;i++) {
    cout<<"======================="<<endl;
    cout<<"CPU "<<i<<endl;
    cout<<"======================="<<endl;

    int olength=20;
    int outvs=2000;
    int embdim=32;

    // Define network
    layer image_in = Input({3,256,256}); //Image
    layer l = image_in;

    l=ReLu(Conv(l,64,{3,3},{2,2}));

    l=ResBlock(l, 64,2,1);//<<<-- output half size
    l=ResBlock(l, 64,2,0);

    l=ResBlock(l, 128,2,1);//<<<-- output half size
    l=ResBlock(l, 128,2,0);

    l=ResBlock(l, 256,2,1);//<<<-- output half size
    l=ResBlock(l, 256,2,0);

    l=ResBlock(l, 512,2,1);//<<<-- output half size
    l=ResBlock(l, 512,2,0);

    l=GlobalAveragePool(l);

    layer lreshape=Reshape(l,{-1});


    // Decoder
    layer ldecin = Input({outvs});
    layer ldec = ReduceArgMax(ldecin,{0});
    ldec = RandomUniform(Embedding(ldec, outvs, 1,embdim,true),-0.05,0.05);

    ldec = Concat({ldec,lreshape});

    l = LSTM(ldec,512,true);

    layer out = Softmax(Dense(l, outvs));

    setDecoder(ldecin);

    model net = Model({image_in}, {out});
    plot(net, "model.pdf");

    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          CS_CPU()
    );


    // Load dataset
    Tensor *x_train=Tensor::zeros({10,3,256,256}); //batch x input_dim
    Tensor *y_train=Tensor::zeros({10,olength,outvs}); //batch x timesteps x ouput_dim

    // to force unrolling
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

    int olength=20;
    int outvs=2000;
    int embdim=32;

    // Define network
    layer image_in = Input({3,256,256}); //Image
    layer l = image_in;

    l=ReLu(Conv(l,64,{3,3},{2,2}));

    l=ResBlock(l, 64,2,1);//<<<-- output half size
    l=ResBlock(l, 64,2,0);

    l=ResBlock(l, 128,2,1);//<<<-- output half size
    l=ResBlock(l, 128,2,0);

    l=ResBlock(l, 256,2,1);//<<<-- output half size
    l=ResBlock(l, 256,2,0);

    l=ResBlock(l, 512,2,1);//<<<-- output half size
    l=ResBlock(l, 512,2,0);

    l=GlobalAveragePool(l);

    layer lreshape=Reshape(l,{-1});


    // Decoder
    layer ldecin = Input({outvs});
    layer ldec = ReduceArgMax(ldecin,{0});
    ldec = RandomUniform(Embedding(ldec, outvs, 1,embdim,true),-0.05,0.05);

    ldec = Concat({ldec,lreshape});

    l = LSTM(ldec,512,true);

    layer out = Softmax(Dense(l, outvs));

    setDecoder(ldecin);

    model net = Model({image_in}, {out});
    plot(net, "model.pdf");

    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          CS_GPU({1})
    );


    // Load dataset
    Tensor *x_train=Tensor::zeros({10,3,256,256}); //batch x input_dim
    Tensor *y_train=Tensor::zeros({10,olength,outvs}); //batch x timesteps x ouput_dim

    // to force unrolling
    fit(net, {x_train}, {y_train}, 10, 1);


    delete x_train;
    delete y_train;
    delete net;

  }
}
