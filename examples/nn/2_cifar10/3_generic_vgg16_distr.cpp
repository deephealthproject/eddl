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
// cifar_vgg16.cpp:
// VGG-16
// Using fit for training
//////////////////////////////////

layer Block1(layer l, int filters) {
    return ReLu(Conv(l, filters,{1, 1},
    {
        1, 1
    }));
}

layer Block3_2(layer l, int filters) {
    l = ReLu(Conv(l, filters,{3, 3},
    {
        1, 1
    }));
    l = ReLu(Conv(l, filters,{3, 3},
    {
        1, 1
    }));
    return l;
}

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


    // download CIFAR data
    //download_cifar10();

    // Settings
    int epochs = testing ? 2 : 32;
    int batch_size = 100;
//    int num_classes = 1000;
    int num_classes = 4;

    // network
    layer in = Input({3, 256, 256});
    //layer in = Input({3, 224, 224});
    layer l = in;


    l = MaxPool(Block3_2(l, 64));
    l = MaxPool(Block3_2(l, 128));
    l = MaxPool(Block1(Block3_2(l, 256), 256));
    l = MaxPool(Block1(Block3_2(l, 512), 512));
    l = MaxPool(Block1(Block3_2(l, 512), 512));

    l = Reshape(l,{-1});
    l = Activation(Dense(l, 512), "relu");

    layer out = Softmax(Dense(l, num_classes));

    // net define input and output layers list
    model net = Model({in},
    {
        out
    });

    

    // Build model
    build(net,
            adam(0.0001), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {
        "categorical_accuracy"
    }, // Metrics
    cs);


    setlogfile(net, "vgg16");

    // plot the model
    if (id == 0)
        plot(net, "model.pdf");

    // get some info from the network
    if (id == 0)
        summary(net);

    // Load and preprocess training data
    // Load and preprocess test data
    /*
    Tensor* x_train = Tensor::load("imagenet/train-images.bi8");
    Tensor* y_train = Tensor::load("imagenet/train-labels.bi8");
    Tensor* x_test = Tensor::load("imagenet/test-images.bi8");
    Tensor* y_test = Tensor::load("imagenet/test-labels.bi8");
    */
    Tensor* x_train ;
    Tensor* y_train ;
    //x_train = Tensor::load("apples/train-images.bi8");
    //        y_train = Tensor::load("apples/train-labels.bi8");
    //         x_train->div_(255.0f);
    Tensor* x_test = Tensor::load("apples/test-images.bin");
    Tensor* y_test = Tensor::load("apples/test-labels.bin");

     x_test->div_(255.0f);

     //if (id==0)
     //y_train->print();
    

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



    // Train
    //fit(net,{x_train},{y_train}, batch_size, epochs);
    // Evaluate
    //evaluate(net,{x_test},{y_test});

    
        for (int i = 0; i < epochs; i++) {
            x_train = Tensor::load("apples/train-images.bi8");
            y_train = Tensor::load("apples/train-labels.bi8");
                x_train->div_(255.0f);
   
            // training, list of input and output tensors, batch, epochs
            fit(net,{x_train},{y_train}, batch_size, 1);
            // Evaluate test
        }
    
    std::cout << "Evaluate test:" << std::endl;
            evaluate(net,{x_test}, {y_test});
        
        

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    end_distributed();

    return EXIT_SUCCESS;
}


///////////
