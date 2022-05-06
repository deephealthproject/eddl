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
    bool use_mpi = true;
    int id;
    char command[100];
    int batch_size = 800;
    int epochs = 10;
    
//    system("export");
//    sprintf(command, "ldd %s",argv[0]); 
//    system(command);    
    
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

    //set_OMP_threads_to_procs_distributed();
    
    // Get MPI process id
    id=get_id_distributed();
    
    // Sync every batch, change every 1 epochs
    //set_avg_method_distributed(AUTO_TIME,8,1);
    set_avg_method_distributed(LIMIT_OVERHEAD,8,0.1);
    
    // Download mnist
    download_mnist();


// Settings
    epochs = (testing) ? 2 : epochs;   
    int num_classes = 10;
    // medical
    //int num_classes = 6;
    //apples
    //int num_classes = 4;

    // Define network
    //layer in = Input({64*64});
    // mnsit
    layer in = Input({1, 28, 28});
    // medical
    // layer in = Input({1, 64, 64});
     // apples
    //  layer in = Input({3, 256, 256});
    layer l = in; // Aux var

    l = Flatten(l);
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 10));

    layer out = Softmax(Dense(l, num_classes), -1); // Softmax axis optional (default=-1)
    model net = Model({in},{out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    if (id == 0) {
       plot(net, "model.pdf");
    }
    
    printf("Available CPUs: %d\n",get_available_CPUs_distributed());
  
      // Define computing service
    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU(get_available_CPUs_distributed());
    } else {
        cs = CS_GPU();
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
   
    /*
    Tensor* x_train = Tensor::load("small/train-images.bin");
    Tensor* y_train = Tensor::load("small/train-labels.bin");
    Tensor* x_test = Tensor::load("small/test-images.bin");
    Tensor* y_test = Tensor::load("small/test-labels.bin");
*/

    /*
    Tensor* x_train = Tensor::load("large/train-images.bin");
    Tensor* y_train = Tensor::load("large/train-labels.bin");
    Tensor* x_test = Tensor::load("large/test-images.bin");
    Tensor* y_test = Tensor::load("large/test-labels.bin");
    */
  
    /*
    Tensor* x_train = Tensor::load("medical/train-images.bin");
    Tensor* y_train = Tensor::load("medical/train-labels.bin");
    Tensor* x_test = Tensor::load("medical/test-images.bin");
    Tensor* y_test = Tensor::load("medical/test-labels.bin");
     */
    /*
    Tensor* x_train = Tensor::load("apples/train-images.bin");
    Tensor* y_train = Tensor::load("apples/train-labels.bin");
    Tensor* x_test = Tensor::load("apples/test-images.bin");
    Tensor* y_test = Tensor::load("apples/test-labels.bin");
    */
   

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

    setlogfile(net,"mnist_mlp_distr");
    
    // Broadcast params 
//    bcast_weights_distributed(net); 
    
    // Train model
    fit(net,{x_train},{y_train}, batch_size, epochs);
    //printf("%f",net->get_accuracy());
    
    // Evaluate
    evaluate(net,{x_test}, {y_test});
    evaluate_distr(net,{x_test}, {y_test});

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
