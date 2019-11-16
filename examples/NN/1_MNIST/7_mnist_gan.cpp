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
// mnist_gan.cpp:
// A very basic GAN for mnist
//////////////////////////////////

layer vreal_loss(vector<layer> in)
{
  // -log( D_out + epsilon )
  return Mult(Log(Sum(in[0],0.0001)),-1);
}

layer vfake_loss(vector<layer> in)
{
  // -log( 1 - D_out + epsilon )
  return Mult(Log(Sum(Diff(1,in[0]),0.0001)),-1);
}


int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Oprimizer

    // Define Generator
    layer gin=GaussGenerator(0.0, 1, {100});
    layer l=gin;

    l=LReLu(Dense(l,256));
    l=LReLu(Dense(l,512));
    l=LReLu(Dense(l,1024));

    layer gout=Tanh(Dense(l,784));

    model gen = Model({gin},{});
    optimizer gopt=sgd(0.01, 0.9);

    build(gen,
          gopt, // Optimizer
          {}, // Losses
          {}, // Metrics
          CS_CPU()
          //CS_GPU({1})
    );


    // Define Discriminator
    layer din=Input({784});
    l = din;
    l = LReLu(Dense(l, 1024));
    l = LReLu(Dense(l, 512));
    l = LReLu(Dense(l, 256));

    layer dout = Sigmoid(Dense(l, 1));

    model disc = Model({din},{});
    optimizer dopt=sgd(0.001, 0.9);
    build(disc,
          dopt, // Optimizer
          {}, // Losses
          {}, // Metrics
          CS_CPU()
          //CS_GPU({1})
    );

    summary(gen);
    summary(disc);

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    // Preprocessing [-1,1]
    eddlT::div_(x_train, 128.0);
    eddlT::sub_(x_train,1.0);


    // Training
    int i,j;
    int num_batches=100;
    int epochs=1000;
    int batch_size = 100;

    tensor batch=eddlT::create({batch_size,784});

// STILL EXPERIMENTAL


    loss rl=newloss(vreal_loss,{dout},"real_loss");
    loss fl=newloss(vfake_loss,{dout},"fake_loss");

    for(i=0;i<epochs;i++) {

      reset_loss(disc);
      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      for(j=0;j<num_batches;j++)  {

        // get a batch from real images
        next_batch({x_train},{batch});
        // generate a batch with generator
        forward(gen,batch_size);

        // Train Discriminator
        zeroGrads(disc);
        // Real
        forward(disc,{batch});
        float dr=compute_loss(rl);
        backward(disc);

        // Fake
        forward(disc,{gout});
        float df=compute_loss(fl);
        backward(disc);

        update(disc);

        // Train Gen
        zeroGrads(gen);
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
