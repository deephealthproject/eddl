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
// mnist_wgan.cpp:
// Wasserstein GAN for mnist
//////////////////////////////////


// loss with a vector of layers
layer vreal_loss(vector<layer> in) //Returns a layer that minimices the loss to maximize for real samples
{
  // maximize for real images (minimize -1 x Value)
  return ReduceMean(Mult(in[0],-1));
}

// OR:
// loss with an unique layer
layer vfake_loss(layer in)         //Returns a layer that maximizes the loss for fake samples.
{
  // minimizes for fake images
  return ReduceMean(in);
}

layer regen_loss(vector<layer> in){
 
	return ReduceMean(Abs(Diff(in[0], in[1])));
}


int main(int argc, char **argv) {

  // Download dataset
  download_mnist();


  // Define Generator
  layer gin=Input({784}); //Gaussian noise generator for creating random input samples
  layer l=gin;

  l=LeakyReLu(Dense(l,1024));
  l=LeakyReLu(Dense(l,512));
  l=LeakyReLu(Dense(l,256));
  l=LeakyReLu(Dense(l,512));
  l=LeakyReLu(Dense(l,1024));

  layer gout=Tanh(Dense(l,784));

  model gen = Model({gin},{}); //Output layer is not specified so it doesn't train automatically. EDDL knows where to get the output because gout is the only node without childs

  // Optimizer
  
  optimizer gopt=rmsprop(0.001);

  build(gen,gopt); // By defatul CS_CPU

  //toGPU(gen); // move toGPU

  summary(gen);

  // Define Discriminator
  layer din=Input({784});
  l = din;
  l = LeakyReLu(Dense(l, 1024));
  l = LeakyReLu(Dense(l, 512));
  l = LeakyReLu(Dense(l, 256));

  layer dout = Dense(l, 1);

  model disc = Model({din},{}); //Output layer is not specified so it doesn't train automatically. EDDL knows where to get the output because gout is the only node without childs

  // Optimizer

  optimizer dopt=rmsprop(0.001);

  build(disc,dopt); // By default CS_CPU

  //toGPU(disc); // move toGPU

  // Define Regenerator
  
  layer rin=Input({784});
  l = rin;
  l = LeakyReLu(Dense(l, 1024));
  l = LeakyReLu(Dense(l, 512));
  l = LeakyReLu(Dense(l, 256));
  l = LeakyReLu(Dense(l, 512));
  l = LeakyReLu(Dense(l, 1024));

  layer rout=Tanh(Dense(l,784));

  model regen = Model({rin},{});

  // Optimizer
  
  optimizer ropt=rmsprop(0.001);

  build(regen,ropt); // By defatul CS_CPU


  summary(disc);

  // Load dataset
  tensor x_train = eddlT::load("trX.bin"); //Loads training samples
  tensor three_train = x_train->select({"18000:24000", ":"});
  tensor eight_train = x_train->select({"48000:54000", ":"});
  //tensor y_train = eddlt::load("trY.bin");

  // Preprocessing [-1,1]
  eddlT::div_(x_train, 128.0);             //Normalizing
  eddlT::sub_(x_train,1.0);                //Normalizing


  // Training
  int i,j;
  int num_batches=1000;
  int epochs=1000;
  int batch_size = 100;

  tensor genBatch=eddlT::create({batch_size,784}); //Create a tensor to hold the batches.
  tensor discBatch=eddlT::create({batch_size,784});


  // Wasserstein GAN params:
  int critic=5;
  float clip=0.01;


  // losses
  loss rl=newloss(vreal_loss,{dout},"real_loss");
  loss fl=newloss(vfake_loss,dout,"fake_loss");
  //loss imgl=new LMeanSquaredError();
  loss imgl=newloss(regen_loss,{rout,gin}, "regen_loss");





  for(i=0;i<epochs;i++) {
    float dr,df;
    fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
    for(j=0;j<num_batches;j++)  {

      for(int k=0;k<critic;k++) {
        // get a batch from real images of number 3
        next_batch({three_train},{genBatch}); //Gets the next batch for training.
        // get a batch from real images of number 8
        next_batch({eight_train},{discBatch}); //Gets the next batch for training.
        // generate a batch with generator

        // Train Discriminator
        zeroGrads(disc); //We set the gradients to zero since they are not automatically zeroed when using backward.
        // Real
        forward(disc,{discBatch}); //Forward the real images through the discriminator to learn them
        dr=compute_loss(rl);   //Compute the loss on the forward
        backward(rl);          //Calculate the gradients with the backward to each layer. Here the discriminator learns to accept real samples.

        // Fake
        forward(disc,detach(forward(gen,{genBatch}))); //Forward the fake images through the discriminator. We detach the output of the forward of our generative net so the gradient doesn't propagate to the generative net
        df=compute_loss(fl);   //We compute the loss on the forward
        backward(fl);          //Calculate the gradients with the backward to each layer. Here the discriminator learns to not accept fake samples.


        update(disc);          //Applies the gradients to each layer on the discriminator.
        clamp(disc,-clip,clip);
      }

      // Train Gen
      zeroGrads(gen);          				 //Puts the grads to 0 to not retrain them
      forward(disc,forward(gen,{genBatch})); //Forwards the output of the generator to the discriminator.
      float gr=compute_loss(rl);             //Compute the result for the generator.
      backward(rl);                          //Calculate the gradients with the backward for each layer.

      update(gen);							 //Applies the gradients to each layer on the generator.

	  zeroGrads(gen);
	  zeroGrads(regen);
	  forward(regen,forward(gen,{genBatch}));
	  float rr=compute_loss(imgl);
	  backward(imgl);


      

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
