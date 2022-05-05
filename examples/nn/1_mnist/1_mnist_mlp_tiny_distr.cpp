/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: 0.9
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
// mnist_mlp.cpp:
// A very basic MLP for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    bool use_mpi = false;
    int id;
    
   // Process arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
         else if (strcmp(argv[i], "--mpi") == 0) use_mpi= true;
    }
    if (use_mpi)
        init_distributed("MPI");
    else 
        init_distributed("NCCL");  

    

    // Init distribuited training
    //id = init_distributed(&argc, &argv);
    id=get_id_distributed();
    
    // Sync every batch, change every 4 epochs
    set_avg_method_distributed(AUTO_TIME,1,4);
    
    // Download mnist
    download_mnist();


    // Settings
    int epochs = (testing) ? 1 : 1;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    //layer in = Input({64*64});
    layer in = Input({1, 28, 28});
    layer l = in; // Aux var

    l = Flatten(l);
    l = LeakyReLu(Dense(l, 10));
 //   l = LeakyReLu(Dense(l, 10));
//    l = LeakyReLu(Dense(l, 10));

    layer out = Softmax(Dense(l, num_classes), -1); // Softmax axis optional (default=-1)
    model net = Model({in},{out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    if (id == 0) {
       plot(net, "model.pdf");
    }
  
  // Define computing service
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
    {
        "categorical_accuracy"
    }, // Metrics
    cs);
    //    toGPU(net, {1}, 100,"low_mem"); // In two gpus, syncronize every 100 batches, low_mem setup

      
   
    // View model
    if (id == 0)
        summary(net);

    // Load dataset
    //Distributed dataset
    //Tensor* x_train = Tensor::load_id("mnist_trX.bin");
    //Tensor* y_train = Tensor::load_id("mnist_trY.bin");
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

   

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":"});
        Tensor* y_mini_train = y_train->select({_range_, ":"});
        Tensor* x_mini_test = x_test->select({_range_, ":"});
        Tensor* y_mini_test = y_test->select({_range_, ":"});

        delete x_train;
        delete y_train;
        delete x_test;
        delete y_test;

        x_train = x_mini_train;
        y_train = y_mini_train;
        x_test = x_mini_test;
        y_test = y_mini_test;
    }
  //y_test->print();
    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);

    //broadcast_CPU_params_distributed(net); 
    //broadcast_GPU_params_distributed(net); 
    
    
    gpu_layer_print (net, 4);
    // Train model
    fit(net,{x_train},{y_train}, batch_size, epochs);
    //printf("%f",net->get_accuracy());
    gpu_layer_print (net, 4);

    // Evaluate
    evaluate(net,{x_test}, {y_test});

    /*
    float acc_goal = 0.98;
    float acc = 0;
    int count=1;
    
   
    while (acc<acc_goal) {
        fit(net,{x_train},{y_train}, batch_size, 1);
        evaluate(net,{x_test}, {y_test});
        if (id==0) 
            acc=net->get_accuracy();
        MPI_Bcast(&acc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        printf("Process %d Iteration: %d Accuracy: %f\n",id, count, acc);
        count++;
    }
    */
    
    // Release objects, layers, optimizer and computing service are released by the net object
    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    // Finalize distributed training
    end_distributed();



    return EXIT_SUCCESS;
}
