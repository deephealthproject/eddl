/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// cifar_conv_da.cpp:
// A very basic Conv for cifar10
// Data augmentation
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv){
    bool testing = false;
    bool use_cpu = false;
    bool use_mpi = false;
    int id=0;
    
    id = init_distributed();
    
   int batch_size = 100;
    int epochs = 10;
  
  // Process arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
        else if (strcmp(argv[i], "--mpi") == 0) use_mpi= true;
        else if (strcmp(argv[i], "--batch-size") == 0) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0) {
            epochs = atoi(argv[++i]);
        }
    }
    if (use_mpi)
      init_distributed("MPI");
    else 
      init_distributed("NCCL");  
    // download CIFAR data
    download_cifar10();

    // Settings
    epochs = testing ? 2 : epochs;
    int num_classes = 10;

    // network
    layer in=Input({3,32,32});
    layer l=in;

    // Data augmentation
    l = RandomHorizontalFlip(l);
    l = RandomCropScale(l, {0.8f, 1.0f});
    l = RandomCutout(l,{0.1,0.5},{0.1,0.5});
//    l = Pad(l, {2, 2});
    ////

    l=MaxPool(ReLu(BatchNormalization(HeUniform(Conv(l,32,{3,3},{1,1},"same",false)))),{2,2});

    l=MaxPool(ReLu(BatchNormalization(HeUniform(Conv(l,64,{3,3},{1,1},"same",false)))),{2,2});

    l=MaxPool(ReLu(BatchNormalization(HeUniform(Conv(l,128,{3,3},{1,1},"same",false)))),{2,2});

    l=MaxPool(ReLu(BatchNormalization(HeUniform(Conv(l,256,{3,3},{1,1},"same",false)))),{2,2});

    l=Reshape(l,{-1});

    l=Activation(BatchNormalization(Dense(l,128)),"relu");

    layer out= Softmax(BatchNormalization(Dense(l, num_classes)));

    // net define input and output layers list
    model net=Model({in},{out});

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU(); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
        // cs = CS_FPGA({1});
    }

    // Build model
    build(net,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs);


    // plot the model
    if (id==0)
        plot(net,"model.pdf");

    // get some info from the network
    if (id==0)        
        summary(net);

    // Load and preprocess training data
    Tensor* x_train = Tensor::load("cifar_trX.bin");
    Tensor* y_train = Tensor::load("cifar_trY.bin");
    x_train->div_(255.0f);

    // Load and preprocess test data
    Tensor* x_test = Tensor::load("cifar_tsX.bin");
    Tensor* y_test = Tensor::load("cifar_tsY.bin");
    x_test->div_(255.0f);

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":"});
        Tensor* y_mini_train = y_train->select({_range_, ":"});
        Tensor* x_mini_test  = x_test->select({_range_, ":"});
        Tensor* y_mini_test  = y_test->select({_range_, ":"});

        delete x_train;
        delete y_train;
        delete x_test;
        delete y_test;

        x_train = x_mini_train;
        y_train = y_mini_train;
        x_test  = x_mini_test;
        y_test  = y_mini_test;
    }

    for(int i=0;i<epochs;i++) {
        // training, list of input and output tensors, batch, epochs
        fit(net,{x_train},{y_train},batch_size, 1);
        // Evaluate train
        std::cout << "Evaluate test:" << std::endl;
        evaluate(net,{x_test},{y_test});
    }

    setlr(net,{0.0001});

    for(int i=0;i<epochs;i++) {
        // training, list of input and output tensors, batch, epochs
        fit(net,{x_train},{y_train},batch_size, 1);
        // Evaluate train
        std::cout << "Evaluate test:" << std::endl;
        evaluate(net,{x_test},{y_test});
    }

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    
    end_distributed();

    return EXIT_SUCCESS;
}
