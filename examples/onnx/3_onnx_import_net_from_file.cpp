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

#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic MLP for mnist
// Using fit for training
//////////////////////////////////


int main(int argc, char **argv) { 
    // Download mnist
    download_mnist();

    // Settings
    int epochs = 1;
    int batch_size = 100;
    int num_classes = 10;

	string path("trained_model.onnx");
	Net* net = import_net_from_onnx_file(path, DEV_CPU);

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}, "low_mem"), // one GPU
          //CS_CPU(), // CPU with maximum threads availables
		  false       // Parameter that indicates that the weights of the net must not be initialized to random values.
    );
	
	//Resize model
	net->resize(batch_size); //Since we don't use "fit", we need to resize the net manually to a correct batch size

    // View model
    summary(net);

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");


    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);

    // Evaluate
    evaluate(net, {x_test}, {y_test});

	return 0;
}


///////////
