/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;


//////////////////////////////////
// Text generation
// Only Decoder
//////////////////////////////////

layer ResBlock(layer l, int filters,int nconv,int half) {
  layer in=l;

  if (half)
      l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{2,2})));
  else
      l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{1,1})));


  for(int i=0;i<nconv-1;i++)
    l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{1,1})));

  if (half)
    return Sum(BatchNormalization(Conv(in,filters,{1,1},{2,2})),l);
  else
    return Sum(l,in);
}


Tensor *onehot(Tensor *in, int vocs)
{
  int n=in->shape[0];
  int l=in->shape[1];
  int c=0;

  Tensor *out=new Tensor({n,l,vocs});
  out->fill_(0.0);

  int p=0;
  for(int i=0;i<n*l;i++,p+=vocs) {
    int w=in->ptr[i];
    if (w==0) c++;
    out->ptr[p+w]=1.0;
  }

  cout<<"padding="<<(100.0*c)/(n*l)<<"%"<<endl;
  return out;
}

int main(int argc, char **argv) {

    download_flickr();

    // Settings
    int epochs = 100;
    int batch_size = 24;

    int olength=20;
    int outvs=2000;
    int embdim=32;

    // Define network
    layer in = Input({3,256,256}); //Image
    layer l = in;

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

    l=Reshape(l,{-1});

    // Decoder
    layer ldec = Input({outvs});
    ldec = ReduceArgMax(ldec,{0});
    ldec = RandomUniform(Embedding(ldec, outvs, 1,embdim),-0.05,0.05);
    l = Decoder(LSTM(ldec,512,true),l,"concat");

    layer out = FullSoftmax(Dense(l, outvs));

    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
    );

    // View model
    summary(net);

    // Load dataset
    Tensor *x_train=Tensor::load("flickr_trX.bin","bin");
    x_train->info(); //1000,256,256,3

    Tensor *xtrain=Tensor::permute(x_train,{0,3,1,2});//1000,3,256,256

    Tensor *y_train=Tensor::load("flickr_trY.bin","bin");
    y_train->info();

    y_train=onehot(y_train,outvs);
    y_train->reshape_({y_train->shape[0],olength,outvs}); //batch x timesteps x input_dim
    y_train->info();

    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {xtrain}, {y_train}, batch_size, 1);
    }




}
