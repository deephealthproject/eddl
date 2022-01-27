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
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "eddl/apis/eddl.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/net/net.h"
#include "eddl/random.h"
#include "eddl/system_info.h"
#include "eddl/utils.h"
#include "eddl/serialization/onnx/eddl_onnx.h"


using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic CNN for mnist
// Using fit for training
//////////////////////////////////

layer Normalization(layer l)
{  
	return l;	
	//return BatchNormalization(l);
}

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download mnist dataset. los .bin para cargar las imagenes de los test y train
    download_cifar10();

    // Settings
    int epochs = (testing) ? 50 : 20;
    int batch_size = 100;
    int num_classes = 10;
    model net;
    layer in, out;
    int dev;

    // Define network mnist
    //layer in = Input({784});
    //layer l = in;  // Aux var

    //l = Reshape(l,{1,28,28});
    //l = MaxPool2D(ReLu(Conv2D(l,32, {3,3},{1,1})),{3,3}, {1,1}, "same");
    //l = MaxPool2D(ReLu(Conv2D(l,64, {3,3},{1,1})),{2,2}, {2,2}, "same");
    //l = MaxPool2D(ReLu(Conv2D(l,128,{3,3},{1,1})),{3,3}, {2,2}, "none");
    //l = MaxPool2D(ReLu(Conv2D(l,256,{3,3},{1,1})),{2,2}, {2,2}, "none");
    //l = Reshape(l,{-1});

    // network cifar10 conv
    // layer in=Input({3,32,32});
    // layer l=in;


    // l=MaxPool(ReLu(Normalization(Conv(l,32,{3,3},{1,1}))),{2,2});
    // l=MaxPool(ReLu(Normalization(Conv(l,64,{3,3},{1,1}))),{2,2});
    // l=MaxPool(ReLu(Normalization(Conv(l,128,{3,3},{1,1}))),{2,2});
    // l=MaxPool(ReLu(Normalization(Conv(l,256,{3,3},{1,1}))),{2,2});

    // l=GlobalMaxPool(l);


    // l=Flatten(l);

    // l=Activation(Dense(l,128),"relu");


    // layer out = Softmax(Dense(l, num_classes));
    //net = Model({in}, {out});


    //net->verbosity_level = 0;

    // dot from graphviz should be installed:
    //plot(net, "model.pdf");
    cout << "antes\n";
    compserv cs = nullptr;
    if (use_cpu) {
        printf("entraa cpu\n");
        cs = CS_CPU();
        printf("mdio cpu\n");
        dev = DEV_CPU;
        printf("sale cpu\n");
    } else {
        printf("entraa gpu\n");
        cs = CS_GPU({1}, "low_mem"); // one GPU
        printf("mdio gpu\n");
        dev = DEV_GPU;  
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
        // cs = CS_FPGA({1});
        printf("sale gpu\n");
    }
    printf("despues");
    // Load weights
    if(testing){
        net = import_net_from_onnx_file("saved-cifar-full.onnx", dev);
	    out = net->lout[0];
    }else{
	//ONNX
        net = import_net_from_onnx_file("model.onnx", dev);
	    out = net->lout[0];
    }
    printf("olaaaaaaaa");
    // Build model, fer un programa que escriga i altre que llisca, en este caso, creo dos modelos nuevos, que se inicializan con pesos diferentes, estaria mal. Hay que usar el mismo modelo
    build(net,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs, false);

    printf("olaaaaaaaa");
    // View model
    summary(net);

    // Load dataset
    Tensor* x_train = Tensor::load("cifar_trX.bin");
    Tensor* y_train = Tensor::load("cifar_trY.bin");

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":"});
        Tensor* y_mini_train = y_train->select({_range_, ":"});

        delete x_train;
        delete y_train;


        x_train = x_mini_train;
        y_train = y_mini_train;

    }

    // Preprocessing
    x_train->div_(255.0f);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Save weights
    if(testing){
        save(net, "saved-weights.bin");
    }else{
        save(net, "saved-yolo-full.bin");
	    save_net_to_onnx_file(net, "saved-yolo-full.onnx");
    }
    


    delete x_train;
    delete y_train;
    delete net;

    return EXIT_SUCCESS;
}
