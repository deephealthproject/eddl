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
#include <chrono>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic MLP for mnist
// Using fit for training
//////////////////////////////////


#define suggest_batch_size(initial_batch, epochs_test, suggested_batch_size) \
    int k; \
    float ratio = 0; \
    float ratio_prev; \
    k=initial_batch; \
    save(net, "saved-weights.bin"); \
    fprintf(stdout, "\nTesting batch size ...\n"); \
    do { \
        load(net, "saved-weights.bin"); \
        ratio_prev = ratio; \
        std::chrono::high_resolution_clock::time_point e1 = std::chrono::high_resolution_clock::now(); \
        fit(net,{x_train}, {y_train}, k, epochs_test); \
        std::chrono::high_resolution_clock::time_point e2 = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration<double> epoch_time_span = e2 - e1; \
        evaluate(net,{x_test},{y_test}); \
        ratio = net->get_accuracy() / epoch_time_span.count(); \
        int id = 0; \
        if (id == 0) { \
            fprintf(stdout, "\nBatch %d: %1.4f secs accuracy %1.4f ratio=%1.4f\n\n", k, epoch_time_span.count(), net->get_accuracy(), ratio); \
        } \
        k = k * 2; \
    } while (ratio > ratio_prev); \
    fprintf(stdout, "\nSuggested batch size is %d\n", k/4); \
    suggested_batch_size=k/4; \

/*
int suggest_batch_size (model net, Tensor* x_train, Tensor* y_train, Tensor* x_test, Tensor* y_test, int initial_batch, int epochs) {
    
    int k;
    float ratio = 0;
    float ratio_prev;
 
    k=initial_batch;
    
     fprintf(stdout, "\nTesting batch size ...\n");
    do {
        ratio_prev = ratio;
        // Train model
        std::chrono::high_resolution_clock::time_point e1 = std::chrono::high_resolution_clock::now();
        fit(net,{x_train}, {y_train}, k, epochs);
        std::chrono::high_resolution_clock::time_point e2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_time_span = e2 - e1;

        ratio = net->get_accuracy() / epoch_time_span.count();

        int id = 0;

        if (id == 0) {
            fprintf(stdout, "\nBatch %d: %1.4f secs accuracy %1.4f ratio=%1.4f\n", k, epoch_time_span.count(), net->get_accuracy(), ratio);
        }
        // Evaluate
        evaluate(net,{x_test},
        {
            y_test
        });
        k = k * 2;
    } while (ratio > ratio_prev);
    fprintf(stdout, "\nSuggested batch size is %d\n", k/4);
    return k/4;
}
 */

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download mnist
    download_mnist();


    // Settings
    int epochs = (testing) ? 2 : 2;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 10));

    layer out = Softmax(Dense(l, num_classes), -1);  // Softmax axis optional (default=-1)
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU({1},"low_mem"); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
        // cs = CS_FPGA({1});
    }

    // Build model
    build(net,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs );
//    toGPU(net, {1}, 100,"low_mem"); // In two gpus, syncronize every 100 batches, low_mem setup

    // View model
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

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);

    //suggest_batch_size(128, 1, batch_size);
 
       
     // Train model
    fit(net,{x_train},{y_train}, batch_size, epochs);
//save(net, "saved-weights.bin");
    //fit(net, {x_train}, {y_train}, 128, 1);
    //load(net, "saved-weights.bin");  
    //fit(net, {x_train}, {y_train}, 256, 1);
    //load(net, "saved-weights.bin");
    //fit(net, {x_train}, {y_train}, 512, 1);

    
    
    // Evaluate
    evaluate(net, {x_test}, {y_test});

    std::vector<Tensor*> preds;
    preds = predict(net, {x_test});
    
    y_test->print();
    
    preds[0]->save("predictions.txt");
    
    
    // Release objects, layers, optimizer and computing service are released by the net object
    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    
    return EXIT_SUCCESS;
}
