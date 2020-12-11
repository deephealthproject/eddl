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
// A very basic CNN for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 3;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Reshape(l,{1,28,28});
	//l = MaxPool(ReLu(Conv(l,32, {3,3},{1,1})),{3,3}, {1,1}, "same");
	//l = MaxPool(ReLu(Conv(l,64, {3,3},{1,1})),{2,2}, {2,2}, "same");
    //l = MaxPool(ReLu(Conv(l,128,{3,3},{1,1})),{3,3}, {2,2}, "none");
    //l = MaxPool(ReLu(Conv(l,256,{3,3},{1,1})),{2,2}, {2,2}, "none");
    //l = ReLu(Conv(l,256,{3,3},{1,1}, "none"));
    l = ReLu(Conv(l,256,{3,3},{1,1}, "none"));
    l = ReLu(Conv(l,256,{3,3},{1,1}, "none"));
    l = ReLu(Conv(l,256,{3,3},{1,1}, "none"));
    l = Reshape(l,{-1});

    layer out = Activation(Dense(l, num_classes),"softmax", {1});
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          adam(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
	  //CS_FPGA({1})
    );

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

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x_test}, {y_test});

	// Export
	string path("trained_model.onnx");
	save_net_to_onnx_file(net, path);

	cout << "Saved net to onnx file" << endl;
	delete net;

	// Import 
	Net* imported_net = import_net_from_onnx_file(path, DEV_CPU);
	
	build(imported_net,
          rmsprop(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
		  CS_GPU({1}, "low_mem"), // one GPU
          //CS_CPU(), // CPU with maximum threads availables
		  false       // Parameter that indicates that the weights of the net must be initialized to random values.
    );

	cout << "Evaluating with imported net" << endl;
    evaluate(imported_net, {x_test}, {y_test}, batch_size);

	string path2("trained_model2.onnx");
	save_net_to_onnx_file(imported_net, path2);


}
