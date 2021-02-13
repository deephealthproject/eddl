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

Tensor *onehot(Tensor *in, int vocs)
{
  int n=in->shape[0];
  int l=in->shape[1];

  Tensor *out=new Tensor({n,l,vocs});
  out->fill_(0.0);

  int p=0;
  for(int i=0;i<n*l;i++,p+=vocs) {
    int w=in->ptr[i];
    out->ptr[p+w]=1.0;
  }

  return out;
}


//////////////////////////////////
// MT
// using EuTrans
//////////////////////////////////
int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download EuTrans
    download_eutrans();

    // Settings
    int epochs = testing ? 1 : 5;
    int batch_size = 32;

    int ilength=30;
    int olength=30;
    int invs=687;
    int outvs=514;
    int embedding=64;

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

    plot(net, "model.pdf");

    optimizer opt=adam(0.01);
    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        //cs = CS_GPU({1}, "low_mem"); // one GPU
        cs = CS_GPU({1}); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
    }

    // Build model
    build(net,
          opt, // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          cs);


    // View model
    summary(net);

    // Load dataset
    Tensor *x_train=Tensor::load("eutrans_trX.bin","bin");
    Tensor *y_train=Tensor::load("eutrans_trY.bin","bin");
    y_train=onehot(y_train,outvs);
    x_train->reshape_({x_train->shape[0],ilength,1}); //batch x timesteps x input_dim
    y_train->reshape_({y_train->shape[0],olength,outvs}); //batch x timesteps x ouput_dim

    Tensor *x_test=Tensor::load("eutrans_tsX.bin","bin");
    Tensor *y_test=Tensor::load("eutrans_tsY.bin","bin");
    y_test=onehot(y_test,outvs);
    x_test->reshape_({x_test->shape[0],ilength,1}); //batch x timesteps x input_dim
    y_test->reshape_({y_test->shape[0],olength,outvs}); //batch x timesteps x ouput_dim

    if (testing) {
        std::string _range_ = "0:" + std::to_string(3 * batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":", ":"});
        Tensor* y_mini_train = y_train->select({_range_, ":", ":"});
        Tensor* x_mini_test  = x_test->select({_range_, ":", ":"});
        Tensor* y_mini_test  = y_test->select({_range_, ":", ":"});

        delete x_train;
        delete y_train;
        delete x_test;
        delete y_test;

        x_train = x_mini_train;
        y_train = y_mini_train;
        x_test  = x_mini_test;
        y_test  = y_mini_test;
    }

    // Train model
    Tensor* ybatch = new Tensor({batch_size, olength,outvs});
    next_batch({y_train},{ybatch});

    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);
    }

    delete net;
    delete x_train;
    delete x_test;
    delete y_train;
    delete y_test;

    return EXIT_SUCCESS;
}
