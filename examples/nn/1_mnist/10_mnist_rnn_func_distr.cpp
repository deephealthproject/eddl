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
// A recurrent NN for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
     int id=0;
    int n_procs;
    
    id = init_distributed();
    n_procs = get_n_procs_distributed();
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download mnist
    download_mnist();

    // Settings
    int epochs = (testing) ? 2 : 5;
    int global_batch_size = 100;
    //int batch_size;
    int num_classes = 10;

        // Define network
    layer in = Input({28});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 32));
    //l = L2(RNN(l, 128, "relu"),0.001);
    l = L2(LSTM(l, 128),0.001);
    layer ls=l;
    l = LeakyReLu(Dense(l, 32));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});


    // dot from graphviz should be installed:
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
          rmsprop(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs);

    // View model
    summary(net);


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

   
        
    int num_batches=x_train->shape[0]/global_batch_size;
    int batches_per_proc=num_batches/n_procs;
    int batch_size=global_batch_size/n_procs;
    
    // Preprocessing
    x_train->div_(255.0);
    x_test->div_(255.0);

    setlogfile(net,"recurrent_mnist");

    Tensor* x_train_batch=new Tensor({batch_size,784});
    Tensor* y_train_batch=new Tensor({batch_size,10});

    // Train model
    //int num_batches=x_train->shape[0]/batch_size;
    for(int i=0;i<epochs;i++) {
        if (id==0)
            printf("Epoch %d\n",i+1);
        reset_loss(net);
       //      for(j=0;j<num_batches;j++)  {
        for (int j = 0; j < batches_per_proc; j++) {
            // get a batch
            next_batch({x_train,y_train},{x_train_batch,y_train_batch});

            x_train_batch->reshape_({batch_size,28,28}); // time x dim
            y_train_batch->reshape_({batch_size,1,10});

            train_batch(net, {x_train_batch}, {y_train_batch});
/*
            zeroGrads(net);
            forward(net,{x_train_batch});
            backward(net,{y_train_batch});
            update(net);

*/
            avg_weights_distributed(net, j, batches_per_proc);

            print_loss(net, j);
            if (id==0)
                printf("\r");

            x_train_batch->reshape_({batch_size,784});

        }
        printf("\n");
    }

    delete x_train_batch;
    delete y_train_batch;

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    
    end_distributed();

    return EXIT_SUCCESS;
}
