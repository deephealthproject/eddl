/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
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
// mnist_mlp_train_batch.cpp:
// A very basic MLP for mnist
// Using train_batch for training
// and eval_batch fot test
//////////////////////////////////

int main(int argc, char **argv) {

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 1;
    int batch_size = 128;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.001, 0.9), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
	      //CS_FPGA({1})
    );

    printf("after build\n");

    // View model
    summary(net);

    setlogfile(net,"mnist");

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    Tensor* xbatch = new Tensor({batch_size, 784});
    Tensor* ybatch = new Tensor({batch_size, 10});

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);


    // Train model
    int i,j;
    tshape s = x_train->getShape();
    int num_batches=s[0]/batch_size;

    for(i=0;i<epochs;i++) {
      reset_loss(net);
      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      for(j=0;j<num_batches;j++)  {

        next_batch({x_train,y_train},{xbatch,ybatch});
        train_batch(net, {xbatch}, {ybatch});

        //OR:
        //vector<int> indices = random_indices(batch_size, s[0]);
        //train_batch(net, {x_train}, {y_train}, indices);

//        print_loss(net,j);
//        printf("\r");

      }
      printf("\n");
    }

    // Print loss and metrics
    vector<float> losses1 = get_losses(net);
    vector<float> metrics1 = get_metrics(net);
    for(int i=0; i<losses1.size(); i++) {
        cout << "Loss: " << losses1[i] << "\t" << "Metric: " << metrics1[i] << "   |   ";
    }
    cout << endl;


    // Evaluate model
    printf("Evaluate:\n");
    s=x_test->getShape();
    num_batches=s[0]/batch_size;

    reset_loss(net);  // Important
    for(j=0;j<num_batches;j++)  {
        vector<int> indices(batch_size);
        for(int i=0;i<indices.size();i++)
          indices[i]=(j*batch_size)+i;

        eval_batch(net, {x_test}, {y_test}, indices);

//        print_loss(net,j);
//        printf("\r");

      }

    // Print loss and metrics
    vector<float> losses2 = get_losses(net);
    vector<float> metrics2 = get_metrics(net);
    for(int i=0; i<losses2.size(); i++) {
        cout << "Loss: " << losses2[i] << "\t" << "Metric: " << metrics2[i] << "   |   ";
    }
    cout << endl;


    //last batch
    if (s[0]%batch_size) {
      int last_batch_size=s[0]%batch_size;
      vector<int> indices(last_batch_size);
      for(int i=0;i<indices.size();i++)
        indices[i]=(j*batch_size)+i;

      eval_batch(net, {x_test}, {y_test}, indices);

//      print_loss(net,j);
//      printf("\r");
    }
    printf("\n");


    // Print loss and metrics
    vector<float> losses3 = get_losses(net);
    vector<float> metrics3 = get_metrics(net);
    for(int i=0; i<losses3.size(); i++) {
        cout << "Loss: " << losses3[i] << "\t" << "Metric: " << metrics3[i] << "   |   ";
    }
    cout << endl;


}
