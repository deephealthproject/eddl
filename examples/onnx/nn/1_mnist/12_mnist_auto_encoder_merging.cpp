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
// mnist_auto_encoder.cpp:
// An autoencoder for mnist
// merging two networs
//////////////////////////////////

int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 1;
    int batch_size = 100;

    // Define encoder
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Activation(Dense(l, 256), "relu");
    l = Activation(Dense(l, 128), "relu");
    layer out = Activation(Dense(l, 64), "relu");

    model encoder = Model({in}, {out});


    // Define decoder
    in = Input({64});
    l = Activation(Dense(in, 128), "relu");
    l = Activation(Dense(l, 256), "relu");

    out = Sigmoid(Dense(l, 784));

    model decoder = Model({in}, {out});


    // Merge both models into a new one
    model net = Model({encoder,decoder});

    // Build model
    build(net,
          adam(0.0001), // Optimizer
          {"mse"}, // Losses
          {"dice"}, // Metrics
          //CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          CS_CPU()
	  //CS_FPGA({1})
    );
    summary(net);
    plot(net, "model.pdf");

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    // Preprocessing
    x_train->div_(255.0f);

    // Train model
    fit(net, {x_train}, {x_train}, batch_size, epochs);

    // Predict with encoder
    vtensor tout=predict(encoder,{x_train});
    tout[0]->info();

	// Evaluate
	evaluate(net, {x_train}, {x_train});
	//delete net;
    //delete encoder;
    //delete decoder;
	// Export
	string path("trained_model.onnx");
	save_net_to_onnx_file(net, path);

	cout << "Saved net to onnx file" << endl;

	// Import 
	Net* imported_net = import_net_from_onnx_file(path, DEV_CPU);
	
	build(imported_net,
          adam(0.001), // Optimizer
          {"mse"}, // Losses
          {"dice"}, // Metrics
		  CS_GPU({1}, "low_mem"), // one GPU
          //CS_CPU(), // CPU with maximum threads availables
		  false       // Parameter that indicates that the weights of the net must be initialized to random values.
    );

	cout << "Evaluating with imported net" << endl;
	evaluate(imported_net, {x_train}, {x_train});


}