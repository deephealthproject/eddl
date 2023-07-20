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
// mnist_rnn.cpp:
// A MLP NN for mnist
// Using fine grained functions
// for training
//////////////////////////////////

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    int id=0;
    int n_procs;
    
    id = init_distributed();
    n_procs = get_n_procs_distributed();
    printf("id %d n_prcas %d \n", id, n_procs);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download mnist
    download_mnist();

    // Settings
    int epochs = (testing) ? 2 : 10;
    int global_batch_size = 128;
    //int batch_size;
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
    if (id == 0)
        plot(net, "model.pdf");

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
          sgd(0.001, 0.9), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs);

    // View model
    if (id == 0)            
        summary(net);

    setlogfile(net,"mnist");

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * global_batch_size);
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
    
    tshape s = x_train->getShape();
    
    int batch_size=global_batch_size/n_procs;    
    int num_batches=s[0]/batch_size;
    int batches_per_proc=num_batches/n_procs;
    


    Tensor* xbatch = new Tensor({batch_size,784});
    Tensor* ybatch = new Tensor({batch_size,10});

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);


    // Train model
    int i,j;

    for (i = 0; i < epochs; i++) {
        reset_loss(net);
        if (id == 0)
            fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs, num_batches);
        //      for(j=0;j<num_batches;j++)  {
        for (j = 0; j < batches_per_proc; j++) {

            next_batch({x_train, y_train},
            {
                xbatch, ybatch
            });
            //train_batch(net, {xbatch}, {ybatch});

            zeroGrads(net);

            forward(net,{xbatch});
            backward(net,{ybatch});
            update(net);

            avg_weights_distributed(net, j, batches_per_proc);

            print_loss(net, j);
            if (id==0)
                printf("\r");
        }
        if (id==0)
            printf("\n");
    }

    delete xbatch;
    delete ybatch;

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    
    end_distributed();    

    return EXIT_SUCCESS;
}