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

int main(int argc, char **argv) {

    // Download dataset
    download_mnist();


    // Define Generator
    layer gin=GaussGenerator(0.0, 1.0, {100});
    layer l=gin;

    l=ReLu(Dense(l,256));
    l=ReLu(Dense(l,512));
    l=ReLu(Dense(l,1024));

    layer gout=Sigmoid(Dense(l,784));

    model gen = Model({gin},{});

    build(gen,
          sgd(0.01, 0.0), // Optimizer
          {}, // Losses
          {}, // Metrics
          //CS_CPU()
          CS_GPU({1})
    );



    // Define Discriminator
    layer din=Input({784});
    l = din;
    l = Dropout(ReLu(Dense(l, 1024)),0.3);
    l = Dropout(ReLu(Dense(l, 512)),0.3);
    l = Dropout(ReLu(Dense(l, 256)),0.3);

    layer dout = Sigmoid(Dense(l, 1));

    model disc = Model({din},{dout});
    build(disc,
            sgd(0.001, 0.0), // Optimizer
          {"mse"}, // Losses
          {"mse"}, // Metrics
          //CS_CPU()
          CS_GPU({1})
    );

    summary(gen);
    summary(disc);

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    // Preprocessing
    eddlT::div_(x_train, 255.0);


    // Training
    int i,j;
    int num_batches=100;
    int epochs=1000;
    int batch_size = 100;

    tensor batch=eddlT::create({batch_size,784});
    tensor real=eddlT::ones({batch_size,1});
    tensor fake=eddlT::zeros({batch_size,1});


    for(i=0;i<epochs;i++) {
      reset_loss(disc);
      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      for(j=0;j<num_batches;j++)  {

        // get random batch
        next_batch({x_train},{batch});

        // Real
        forward(disc,{batch});

        reset_grads(disc);
        backward(disc,{real});
        update(disc);


        compute_loss(disc);
        print_loss(disc,j);
        printf("\r");


        // Fake
        forward(gen,batch_size);
        forward(disc,{getTensor(gout)});

        reset_grads(disc);
        backward(disc,{fake});
        update(disc);

        compute_loss(disc);
        print_loss(disc,j);
        printf("\r");

        // Update Gen
        forward(disc,{getTensor(gout)});

        reset_grads(disc);
        backward(disc,{real});

        reset_grads(gen);
        copyTensor(getGrad(din),getGrad(gout));

        backward(gen);
        update(gen);

      }
      printf("\n");

      // Generate some num_samples
      forward(gen,batch_size);

      tensor output=getTensor(gout);

      tensor img=eddlT::select(output,0);
      eddlT::reshape_(img,{1,1,28,28});
      eddlT::save_png(img,"img.png");

      tensor img1=eddlT::select(output,5);
      eddlT::reshape_(img1,{1,1,28,28});
      eddlT::save_png(img1,"img1.png");

      tensor img2=eddlT::select(batch,0);
      eddlT::reshape_(img2,{1,1,28,28});
      eddlT::save_png(img2,"img2.png");

      tensor img3=eddlT::select(batch,5);
      eddlT::reshape_(img3,{1,1,28,28});
      eddlT::save_png(img3,"img3.png");

    }





}


///////////
