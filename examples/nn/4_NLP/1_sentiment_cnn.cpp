/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
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
// Embeding+CNN
// using imdb preprocessed from keras
// 2000 words
//////////////////////////////////

int main(int argc, char **argv) {
  // Download Imdb
  download_imdb_2000();

  // Settings
  int epochs = 10;
  int batch_size = 32;

  int length=250;
  int embdim=32;
  int vocsize=2000;

  // Define network
  layer in = Input({length});
  layer l = in;

  layer lE = RandomUniform(Embedding(l, vocsize, length,embdim),-0.05,0.05);
  l = Reshape(lE,{1,length,embdim});

  layer l1 = ReLu(BatchNormalization(Conv(l,128,{1,embdim},{1,1},"same,none",false)));
  layer l2 = ReLu(BatchNormalization(Conv(l,128,{2,embdim},{1,1},"same,none",false)));
  layer l3 = ReLu(BatchNormalization(Conv(l,128,{3,embdim},{1,1},"same,none",false)));

  layer lc1=l=Concat({l1,l2,l3});

  l=GlobalMaxPool(l);

  l = Reshape(l,{-1});

  l = ReLu(BatchNormalization(Dense(l,256)));

  layer out = Sigmoid(Dense(l, 1));
  model net = Model({in}, {out});

  // dot from graphviz should be installed:
  plot(net, "model.pdf");

  // Build model
  build(net,
    adam(0.0001), // Optimizer
    {"cross_entropy"}, // Losses
    {"binary_accuracy"}, // Metrics
    //CS_GPU({1}) // one GPU
    //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
    CS_CPU()
  );
  // View model
  summary(net);

  // Load dataset
  Tensor* x_train=Tensor::load("imdb_2000_trX.bin");
  Tensor* y_train=Tensor::load("imdb_2000_trY.bin");

  Tensor* x_test=Tensor::load("imdb_2000_tsX.bin");
  Tensor* y_test=Tensor::load("imdb_2000_tsY.bin");


  // Train model
  for(int i=0;i<epochs;i++) {
    fit(net, {x_train}, {y_train}, batch_size, 1);
    evaluate(net,{x_test},{y_test});
  }



}
