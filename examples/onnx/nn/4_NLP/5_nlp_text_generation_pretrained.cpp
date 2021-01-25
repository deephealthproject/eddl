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
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed


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

    string path("resnet18-v1-7.onnx");
    // Download from:
    // https://www.dropbox.com/s/tn0d87dr035yhol/resnet18-v1-7.onnx

  	Net* net_onnx = import_net_from_onnx_file(path, DEV_CPU);

    // Remove last layer
    removeLayer(net_onnx, "resnetv15_dense0_fwd");

    layer in=getLayer(net_onnx, "data");
    layer flatten=getLayer(net_onnx, "flatten_170");

    // Decoder
    layer ldec = Input({outvs});
    ldec = ReduceArgMax(ldec,{0});
    ldec = RandomUniform(Embedding(ldec, outvs, 1,embdim),-0.05,0.05);
    layer l = Decoder(LSTM(ldec,512,true),flatten,"concat");

    layer out = Softmax(Dense(l, outvs));

    model net = Model({in}, {out});


    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
          ,false  //initialize false
    );

    // View model
    summary(net);
    plot(net, "model.pdf");


    // Load dataset
    Tensor *x_train=Tensor::load("flickr_trX.bin","bin");
    x_train->info();

    Tensor *xtrain=Tensor::permute(x_train,{0,3,1,2});
    xtrain->info();

    delete x_train;
    x_train=Tensor::zeros({1000, 3, 224, 224});

    Tensor::scale(xtrain,x_train,{224,224});
    x_train->info();
    x_train->div_(255.0);

    Tensor *y_train=Tensor::load("flickr_trY.bin","bin");
    y_train->info();

    y_train=onehot(y_train,outvs);
    y_train->reshape_({y_train->shape[0],olength,outvs}); //batch x timesteps x input_dim
    y_train->info();


    setTrainable(net,"flatten_170",false);
    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);
    }

}
