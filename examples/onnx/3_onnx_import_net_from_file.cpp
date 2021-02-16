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
    // Download cifar
    download_cifar10();

    // Settings
    int epochs = 1;
    int batch_size = 100;
    int num_classes = 10;
    vector<string> names;

/*
    string path("resnet18.onnx");
	  Net* net = import_net_from_onnx_file(path, DEV_CPU);
    
    for(auto l:net->layers) 
      cout<<l->name<<endl;

    getchar();

    // Build model
    build(net,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}, "low_mem"), // one GPU
          //CS_CPU(), // CPU with maximum threads availables
		  false       // Parameter that indicates that the weights of the net must not be initialized to random values.
    );

     // View model
    summary(net);
    getchar();  
    */

    Net* net=download_resnet18(true,{3, 32, 32});  
    // true: remove last layers and set new top=flatten 
    // new input_size {3,32,32} from {224,224,3}

    layer l=getLayer(net,"top");
    layer out=Softmax(Dense(l,10,true,"newdense")); // true is for the bias.

    // create a new model from input output
    layer in=getLayer(net,"input");

    net=Model({in},{out});

    // Build model
    build(net,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}), // one GPU
          //CS_CPU(), // CPU with maximum threads availables
		  false       // Parameter that indicates that the weights of the net must not be initialized to random values.
    );

     // View model
    summary(net);
  

    // force initialization of new layers
    initializeLayer(net,"newdense");
    

    // Load and preprocess training data
    Tensor* x_train = Tensor::load("cifar_trX.bin");
    Tensor* y_train = Tensor::load("cifar_trY.bin");
    x_train->div_(255.0f);

    // Load and preprocess test data
    Tensor* x_test = Tensor::load("cifar_tsX.bin");
    Tensor* y_test = Tensor::load("cifar_tsY.bin");
    x_test->div_(255.0f);
  

    for(int i=0;i<net->layers.size();i++) {
      cout<<net->layers[i]->name<<"  "<<net->vfts[i]->name<<endl;
    }

    // names of layers pretrained
    for(auto l:net->layers) { 
      if (l->name!="top") names.push_back(l->name);
      else break;
    }

    //Train few epochs frozen  
    for(auto n:names)
      setTrainable(net,n,false);
    

    fit(net,{x_train},{y_train},batch_size, 10);

    // unfreeze
    for(auto n:names)
      setTrainable(net,n,true);

    //Train few epochs all layers
    fit(net,{x_train},{y_train},batch_size, 10);


    // Evaluate
    evaluate(net, {x_test}, {y_test},100);

	return 0;
}


///////////
