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

#include "eddl/apis/eddl.h"

using namespace eddl;

//////////////////////////////////
// mnist_gan.cpp:
// A basic enc-dec with
// user defined loss
//////////////////////////////////

layer mse_loss(vector<layer> in)
{
  layer diff=Diff(in[0],in[1]);
  return Mult(diff,diff);
}

layer dice_loss(vector<layer> in)
{
  layer num=ReduceMean(Mult(in[0],in[1]),{0,1,2});
  layer den=ReduceMean(Sum(in[0],in[1]),{0,1,2});

  return Diff(1.0,Div(num,den));
}

int main(int argc, char **argv) {
    int i,j;

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 100;
    int batch_size = 100;


    // Define network
    layer in = Input({784});
    layer target = Reshape(in,{1,28,28});
    layer l = in;  // Aux var

    l = Reshape(l,{1,28,28});
    l = ReLu(Conv(l,8,{3,3}));
    l = ReLu(Conv(l,16,{3,3}));
    l = ReLu(Conv(l,8,{3,3}));
    layer out = Sigmoid(Conv(l,1,{3,3}));
    model net = Model({in}, {});
    // Build model
    build(net,adam(0.001));
    summary(net);
    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");

    // Preprocessing
    x_train->div_(255.0f);

    loss mse=newloss(mse_loss,{out,target},"mse_loss");
    loss dice=newloss(dice_loss,{out,target},"dice_loss");

    Tensor *batch=new Tensor({batch_size,784});
    int num_batches=x_train->shape[0]/batch_size;
    for(i=0;i<epochs;i++) {

      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      float val=0.0;
      float mseloss=0;
      for(j=0;j<num_batches;j++)  {

        next_batch({x_train},{batch});

        zeroGrads(net);

        forward(net,{batch});

        val+=compute_loss(dice)/batch_size;

        cout<<"diceloss="<<val/(j+1)<<"\r";
        fflush(stdout);

        backward(dice);

        update(net);


      }

      printf("\n");


    }





}


///////////
