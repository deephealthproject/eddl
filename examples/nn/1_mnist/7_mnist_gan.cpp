/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
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
  return ReduceMean(Mult(Log(Sum(Diff(1,in[0]),0.0001)),-1));
}


int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Oprimizer

    // Define Generator
    layer gin=GaussGenerator(0.0, 1, {25});
    layer l=gin;

    l=LeakyReLu(Dense(l,256));
    l=LeakyReLu(Dense(l,512));
    l=LeakyReLu(Dense(l,1024));

    layer gout=Tanh(Dense(l,784));

    model gen = Model({gin},{});
    optimizer gopt=adam(0.0001);

    build(gen,gopt); // CS_CPU by default
    //toGPU(gen); // GPU {1} by default


    // Define Discriminator
    layer din=Input({784});
    l = din;
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 512));
    l = LeakyReLu(Dense(l, 256));

    layer dout = Sigmoid(Dense(l, 1));

    model disc = Model({din},{});
    optimizer dopt=adam(0.0001);

    build(disc,dopt); // CS_CPU by default
    //toGPU(disc); // GPU {1} by default

    summary(gen);
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

    loss rl=newloss(vreal_loss,{dout},"real_loss");
    loss fl=newloss(vfake_loss,{dout},"fake_loss");

    for(i=0;i<epochs;i++) {

      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      float dr,df,gr;
      for(j=0;j<num_batches;j++)  {

        // get a batch from real images
        next_batch({x_train},{batch});

        // Train Discriminator
        zeroGrads(disc);
        // Real
        forward(disc,{batch});
        float dr=compute_loss(rl);
        backward(rl);


        // Fake
        forward(disc,detach(forward(gen,batch_size)));
        float df=compute_loss(fl);
        backward(fl);
        update(disc);

        // Train Gen
        zeroGrads(gen);
        forward(disc,forward(gen,batch_size));
        float gr=compute_loss(rl);
        backward(rl);

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
