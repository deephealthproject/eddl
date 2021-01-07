/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
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
    int epochs = 0;
    int batch_size = 24;

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
    ldec = RandomUniform(Embedding(ldec, outvs, 1,embdim),-0.05,0.05);

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
          //CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          CS_CPU()
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


    /////////////////////////////////////////////
    // INFERENCE
    /////////////////////////////////////////////




    /////////////////////////////////////////////
    /// Get all the reshapes of the images
    /// Only use the CNN
    /////////////////////////////////////////////

    Tensor *timage=new Tensor({x_train->shape[0], 512}); //images reshape

    model cnn=Model({image_in},{lreshape});


    build(cnn,
          adam(0.001), // not relevant
          {"mse"}, // not relevant
          {"mse"}, // not relevant
          CS_CPU() // CPU
    );
    summary(cnn);
    plot(cnn,"cnn.pdf");

    // forward images
    Tensor* xbatch = new Tensor({batch_size,3,256,256});

    int numbatches=x_train->shape[0]/batch_size;
    for(int j=0;j<1;j++)  {
        cout<<"batch "<<j<<endl;

        next_batch({x_train},{xbatch});
        forward(cnn,{xbatch});

        Tensor* ybatch=getOutput(lreshape);

        string sample=to_string(j*batch_size)+":"+to_string((j+1)*batch_size);
        timage->set_select({sample,":"},ybatch);


        delete ybatch;
    }



    /////////////////////////////////////////////
    /// Create Decoder non recurrent for n-best
    /////////////////////////////////////////////

    ldecin = Input({outvs});
    layer image = Input({512});
    //layer lstates = States({2,512});

    ldec = ReduceArgMax(ldecin,{0});
    ldec = RandomUniform(Embedding(ldec, outvs, 1,embdim),-0.05,0.05);

    ldec = Concat({ldec,image});

    l = LSTM(ldec,512,true);

    l->isrecurrent=false; // Important

    out = Softmax(Dense(l, outvs));

    model decoder=Model({ldecin,image},{out});

    // Build model
    build(decoder,
          adam(0.001), // not relevant
          {"softmax_cross_entropy"}, // not relevant
          {"accuracy"}, // not relevant
          CS_CPU() // CPU
    );

    // View model
    summary(decoder);
    plot(decoder, "decoder.pdf");

    // Copy params from trained net
    copyParam(getLayer(net,"LSTM1"),getLayer(decoder,"LSTM2"));
    copyParam(getLayer(net,"dense1"),getLayer(decoder,"dense2"));
    copyParam(getLayer(net,"embedding1"),getLayer(decoder,"embedding2"));


   ////// N-best for sample s
   int s=100; //sample 100
   Tensor *treshape=timage->select({to_string(s),":"});
   Tensor *text=y_train->select({to_string(s),":",":"}); //1 x olength x outvs
   Tensor *state=Tensor::zeros({512});

   for(int j=0;j<olength;j++) {

     Tensor *word;
     if (j==0) word=Tensor::zeros({1,outvs});
     else {
       string n=to_string(j-1);
       word=text->select({"0",n,":"});
       word->reshape_({1,1,outvs});
     }

     //setState(lstate,state)
     treshape->reshape_({1,512});

     cout<<"forward"<<endl;
     forward(decoder,(vtensor){word,treshape});

     Tensor *outword=getOutput(out);
     //delete state;
     //state=getState(lstate);
   }

}
