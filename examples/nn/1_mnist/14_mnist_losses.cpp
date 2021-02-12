/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

#include "eddl/apis/eddl.h"

using namespace eddl;

//////////////////////////////////
// mnist_losses.cpp:
// A basic enc-dec with
// user defined loss
//////////////////////////////////


// l2_loss
layer mse_loss(vector<layer> in)
{
  layer diff=Sub(in[0],in[1]);
  return Mult(diff,diff);
}

// l1_loss
layer l1_loss(vector<layer> in)
{
  return Abs(Sub(in[0],in[1]));
}


// Dice loss image-level
layer dice_loss_img(vector<layer> in)
{
  layer num=Mult(2,ReduceSum(Mult(in[0],in[1]),{0,1,2}));
  layer den=ReduceSum(Add(in[0],in[1]),{0,1,2});

  return Sub(1.0,Div(num,den));
}

// Dice loss pixel-level
layer dice_loss_pixel(vector<layer> in)
{
  layer num=Mult(2,ReduceSum(Mult(in[0],in[1]),{0}));
  layer den=ReduceSum(Add(in[0],in[1]),{0});

  num=Add(num,1);
  den=Add(den,1);

  return Sub(1.0,Div(num,den));
}

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }
    int i,j;

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 1;
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

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU({1}, "low_mem"); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
        // cs = CS_FPGA({1});
    }
    // Build model
    build(net,
          adam(0.001), // Optimizer
          {}, // Losses
          {}, // Metrics
          cs);

    summary(net);
    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":"});

        delete x_train;

        x_train = x_mini_train;
    }

    // Preprocessing
    x_train->div_(255.0f);

    loss mse=newloss(mse_loss,{out,target},"mse_loss");
    loss dicei=newloss(dice_loss_img,{out,target},"dice_loss_img");
    loss dicep=newloss(dice_loss_pixel,{out,target},"dice_loss_pixel");

    Tensor *batch=new Tensor({batch_size,784});
    int num_batches=x_train->shape[0]/batch_size;
    for(i=0;i<epochs;i++) {

      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      float diceploss=0.0;
      float diceiloss=0.0;
      float mseloss=0;

      for(j=0;j<num_batches;j++)  {

        cout<<"Batch "<<j;
        next_batch({x_train},{batch});

        zeroGrads(net);

        forward(net,{batch});

        diceploss+=compute_loss(dicep)/(float)batch_size;
        cout<<" ( dice_pixel_loss="<< std::setprecision(3) << std::fixed << diceploss/(float)(j+1);

        diceiloss+=compute_loss(dicei)/(float)batch_size;
        cout<<"; dice_img_loss="<< std::setprecision(3) << std::fixed << diceiloss/(float)(j+1);

        mseloss+=compute_loss(mse)/(float)batch_size;
        cout<<"; mse_loss=" << std::setprecision(3) << std::fixed <<  mseloss/(float)(j+1);

        cout <<" )" <<"\r";
        fflush(stdout);

        optimize(dicep);
        //optimize({dicei,mse});


        update(net);
      }

      printf("\n");


    }

    delete dicep;
    delete dicei;
    delete mse;

    delete batch;

    delete x_train;
    delete net;
    
    return EXIT_SUCCESS;
}
