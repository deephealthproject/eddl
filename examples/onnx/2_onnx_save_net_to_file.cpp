/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"
#include "eddl/apis/eddlT.h"
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
    int batch_size = 1000;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

	 l = l=Reshape(l,{1,28,28});
	 l=MaxPool(ReLu(BatchNormalization(Conv(l,32,{3,3},{1,1}))),{2,2});
	 l=MaxPool(ReLu(BatchNormalization(Conv(l,64,{3,3},{1,1}))),{2,2});
	 l=MaxPool(ReLu(BatchNormalization(Conv(l,128,{3,3},{1,1}))),{2,2});
					 
	 l=Flatten(l);
									 
	 l=Activation(Dense(l,256),"relu");
											 
	 l=Activation(Dense(l,128),"relu");
													 
	 layer out=Activation(Dense(l,num_classes),"softmax");
															 
	 // Net define input and output layers list
	 model net=Model({in},{out});// Build model

    build(net,
          rmsprop(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
		  CS_GPU({1}, "low_mem"), // one GPU
          //CS_CPU(), // CPU with maximum threads availables
		  true       // Parameter that indicates that the weights of the net must be initialized to random values.
    );
	
	

    // View model
    summary(net);

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");


    // Preprocessing
    eddlT::div_(x_train, 255.0);
    eddlT::div_(x_test, 255.0);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x_test}, {y_test});

	// Export
	string path("trained_model.onnx");
	save_net_to_onnx_file(net, path);

	cout << "Saved net to onnx file" << endl;
	
	return 0;
}


///////////
