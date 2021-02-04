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

#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic MLP for mnist
// Using fit for training
//////////////////////////////////


int main(int argc, char **argv) { 
	/* 2020-01-10
#define layer Layer*
#define model Net*
*/
    // Download mnist
    download_mnist();

    // Settings
    int epochs = 1;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var
	/*
  	l=Reshape(l,{1,28, 28});
  	l=MaxPool(ReLu(Conv(l,32,{3,3},{1,1})),{2,2});
  	l=MaxPool(ReLu(Conv(l,32,{3,3},{1,1})),{2,2});
	*/
  	l=Reshape(l,{-1});
    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

	net->enable_distributed();

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}) // one GPU
          CS_CPU(4), // CPU with maximum threads availables
		  true
    );

	void* serialized_net;
	size_t model_size = serialize_net_to_onnx_pointer(net, serialized_net, false );

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

	Tensor* x_mini_train = x_train->select({"0:6000",":"});
	Tensor* y_mini_train = y_train->select({"0:6000",":"});

	//resize_model(net, batch_size);
    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);
    //fit(net, {x_mini_train}, {y_mini_train}, batch_size, epochs);

	// Going to reset gradients
	//cout << "reseting gradients" << endl;
	//net->reset_accumulated_gradients();
	//cout << "reset succesful" << endl;

    // Evaluate
	cout << "Evaluating test before import" << endl;
    evaluate(net, {x_test}, {y_test});

	

	string path("mnist.onnx");
	// Export
	//save_net_to_onnx_file(net, path);
	void* serialized_gradients;
	size_t gradients_size = serialize_net_to_onnx_pointer(net, serialized_gradients, true );
	//std::string* model_string = serialize_net_to_onnx_string(net);
	
	cout << "Exported gradients" << endl;

	void * serialized_net_once_trained;
	size_t snot_size = serialize_net_to_onnx_pointer( net, serialized_net_once_trained, false );



	LConv::reset_name_counter();
	LDense::reset_name_counter();
	/*
    // Define network
    in = Input({784});
    l = in;  // Aux var
  	l=Reshape(l,{1,28, 28});

  	l=MaxPool(ReLu(Conv(l,32,{3,3},{1,1})),{2,2});
  	l=MaxPool(ReLu(Conv(l,32,{3,3},{1,1})),{2,2});

  	l=Reshape(l,{-1});

    l = ReLu(Dense(l, 1024));

    out = Softmax(Dense(l, num_classes));
    model net2 = Model({in}, {out});

	*/
	
	// Import
	Net* imported_net = import_net_from_onnx_pointer(serialized_net, model_size);
	//Net* imported_net = import_net_from_onnx_file(path);
	//Net* imported_net = import_net_from_onnx_string(model_string);


    // Build model
    build(imported_net,
          rmsprop(0.01), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}) // one GPU
          CS_CPU(4), // CPU with maximum threads availables
		  false
    );

	//resize_model(imported_net, batch_size);

    // View model
    summary(imported_net);

	/*
	for( int i = 0; i < 10; i++)
		cout << "W: First value of W in first dense layer " << ((LDense *)(imported_net->layers[4]))->W->ptr[i] << endl;
	*/
	cout << "net  |layers| = " << net->layers.size() << endl << flush;
	cout << "net' |layers| = " << imported_net->layers.size() << endl << flush;
	for( auto layer : net->layers ) {
		cout << layer->name << "  " ;
		if ( LDense *t = dynamic_cast<LDense*>( layer ) ) {
			cout << " sum(W): " << t->W->sum();
		}
		cout << endl << flush;
	}
	for( auto layer : imported_net->layers ) {
		cout << layer->name << "  " ;
		if ( LDense *t = dynamic_cast<LDense*>( layer ) ) {
			cout << " sum(W): " << t->W->sum();
		}
		cout << endl << flush;
	}

    // Evaluate
	cout << "Evaluating test before new weights" << endl;
    evaluate(imported_net, {x_test}, {y_test});

	// Apply grads
	//apply_grads_from_onnx_pointer( imported_net, serialized_gradients, gradients_size );
	
	/*
	for( int i = 0; i < 10; i++)
		cout << "W: First value of W in first dense layer " << ((LDense *)(imported_net->layers[4]))->W->ptr[i] << endl;
	*/
	cout << "net  |layers| = " << net->layers.size() << endl << flush;
	cout << "net' |layers| = " << imported_net->layers.size() << endl << flush;
	for( auto layer : net->layers ) {
		cout << layer->name << "  " ;
		if ( LDense *t = dynamic_cast<LDense*>( layer ) ) {
			cout << " sum(W): " << t->W->sum();
		}
		cout << endl << flush;
	}
	for( auto layer : imported_net->layers ) {
		cout << layer->name << "  " ;
		if ( LDense *t = dynamic_cast<LDense*>( layer ) ) {
			cout << " sum(W): " << t->W->sum();
		}
		cout << endl << flush;
	}

	cout << "Evaluating test after applying gradients" << endl;
    evaluate(imported_net, {x_test}, {y_test});
	
	// Set new weights
	//set_weights_from_onnx_pointer( imported_net, serialized_net_once_trained, snot_size );
    // Evaluate
	cout << "Evaluating test after new weights" << endl;
    evaluate(imported_net, {x_test}, {y_test});

	for( int k=0; k < 10; k++ ) {
		// Train
		fit(net, {x_train}, {y_train}, batch_size, epochs);
		// Evaluate
		cout << "Evaluating test after training" << endl;
		evaluate(net, {x_test}, {y_test});

		// Generate gradients
		gradients_size = serialize_net_to_onnx_pointer( net, serialized_gradients, true );
		net->reset_accumulated_gradients();

		// Apply gradients to imported_net
		//apply_grads_from_onnx_pointer( imported_net, serialized_gradients, gradients_size );

		// Evaluate imported net
		evaluate( imported_net, {x_test}, {y_test});
	}

	return 0;
/*
	string path2("mnist2.onnx");
	//saveModelToOnnx(imported_net,path2);

	string path3("mnist3.onnx");
	//Net* imported_net2 = import_net_from_onnx_file(path2);


    // Build model
    build(imported_net2,
          rmsprop(0.01), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}) // one GPU
          CS_CPU(), // CPU with maximum threads availables
		  false
    );
	resize_model(imported_net2, batch_size);
	saveModelToOnnx(imported_net2, path3);

	return 0;
	*/

}


///////////
