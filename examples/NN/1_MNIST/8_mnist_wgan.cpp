/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace eddl;

//////////////////////////////////
// mnist_wgan.cpp:
// Wasserstein GAN for mnist
//////////////////////////////////

// loss with an vector of layers
layer vreal_loss(vector<layer> in)
{
  // maximize for real images (mimize -1 x Value)
  return Mult(in[0],-1);
}

// OR:
// loss with an unique layer
layer vfake_loss(layer in)
{
  // minimizes for fake images
  return in;
}


int main(int argc, char **argv) {

  // Download dataset
  download_mnist();


  // Define Generator
  layer gin=GaussGenerator(0.0, 1, {100});
  layer l=gin;

  l=LReLu(Dense(l,256));
  l=LReLu(Dense(l,512));
  l=LReLu(Dense(l,1024));

  layer gout=Tanh(Dense(l,784));

  model gen = Model({gin},{});
  optimizer gopt=rmsprop(0.001);

  build(gen,gopt); // By defatul CS_CPU

  toGPU(gen); // move toGPU

  summary(gen);


  // Define Discriminator
  layer din=Input({784});
  l = din;
  l = LReLu(Dense(l, 1024));
  l = LReLu(Dense(l, 512));
  l = LReLu(Dense(l, 256));

  layer dout = Dense(l, 1);

  model disc = Model({din},{});
  optimizer dopt=rmsprop(0.001);

  build(disc,dopt); // By defatul CS_CPU

  toGPU(disc); // move toGPU

  summary(disc);


  // Load dataset
  tensor x_train = eddlT::load("trX.bin");
  // Preprocessing [-1,1]
  eddlT::div_(x_train, 128.0);
  eddlT::sub_(x_train,1.0);


  // Training
  int i,j;
  int num_batches=1000;
  int epochs=1000;
  int batch_size = 100;

  tensor batch=eddlT::create({batch_size,784});


  // Wasserstein GAN params:
  int critic=5;
  float clip=0.01;

  // losses
  loss rl=newloss(vreal_loss,{dout},"real_loss");
  loss fl=newloss(vfake_loss,dout,"fake_loss");


  for(i=0;i<epochs;i++) {
    float dr,df;
    fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
    for(j=0;j<num_batches;j++)  {

      for(int k=0;k<critic;k++) {
        // get a batch from real images
        next_batch({x_train},{batch});
        // generate a batch with generator
        forward(gen,batch_size);

        // Train Discriminator
        zeroGrads(disc);
        // Real
        forward(disc,{batch});
        dr=compute_loss(rl);
        backward(disc);

        // Fake
        forward(disc,{gout});
        df=compute_loss(fl);
        backward(disc);

        update(disc);

        clamp(disc,-clip,clip);
      }

      // Train Gen
      zeroGrads(gen);
      forward(gen,batch_size);
      forward(disc,{gout});
      float gr=compute_loss(rl);
      backward(disc);
      copyGrad(din,gout);
      backward(gen);
      update(gen);

      printf("Batch %d -- Total Loss=%1.3f  -- Dr=%1.3f  Df=%1.3f  Gr=%1.3f\r",j+1,dr+df+gr,dr,df,gr);

      fflush(stdout);

    }

    printf("\n");

    // Generate some num_samples
    forward(gen,batch_size);
    tensor output=getTensor(gout);

    tensor img=eddlT::select(output,0);

    eddlT::reshape_(img,{1,1,28,28});
    eddlT::save(img,"img.png","png");

    tensor img1=eddlT::select(output,5);
    eddlT::reshape_(img1,{1,1,28,28});
    eddlT::save(img1,"img1.png","png");


  }





}


///////////
