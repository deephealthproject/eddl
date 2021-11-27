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
// A very basic CNN for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    int id;

        
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }
    
    // Define computing service
    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else { 
	cs=CS_MPI_DISTRIBUTED();
    }

    
    // Init distribuited training
    id = get_id_distributed();
    
    // Sync every batch, change every 2 epochs
    set_method_distributed(AUTO_TIME,1,2);

    // Download mnist
    download_mnist();

    // Settings
    int epochs = (testing) ? 2 : 10;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in; // Aux var

    l = Reshape(l,{1, 28, 28});
    l = MaxPool2D(ReLu(Conv2D(l, 32,{3, 3},
    {
        1, 1
    })),{3, 3},
    {
        1, 1
    }, "same");
    //    l = MaxPool2D(ReLu(Conv2D(l,64, {3,3},{1,1})),{2,2}, {2,2}, "same");
    //    l = MaxPool2D(ReLu(Conv2D(l,128,{3,3},{1,1})),{3,3}, {2,2}, "none");
    //    l = MaxPool2D(ReLu(Conv2D(l,256,{3,3},{1,1})),{2,2}, {2,2}, "none");
    l = Reshape(l,{-1});

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in},
    {
        out
    });
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    

    // Build model
    build(net,
            rmsprop(0.001), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {
        "categorical_accuracy"
    }, // Metrics
    cs);

    // View model
    if (id == 0)
        summary(net);

    // Load dataset
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

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);

    // Train model
    fit(net,{x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    evaluate(net,{x_test},
    {
        y_test
    });

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    // Finalize distributed training
    end_distributed();

    return EXIT_SUCCESS;
}
