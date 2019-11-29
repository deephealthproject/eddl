#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace eddl;


layer FMP(layer l)
{
  return MaxPool(l, {2,2},{1,1},"same");
}

layer Block(layer l,int filters, vector<int> kernel, vector<int> stride)
{
  return MaxPool(ReLu(Conv(l,filters, kernel,stride)));
}

int main(int argc, char** argv)
{

  // download MNIST data
    download_mnist();

    // Settings
    int epochs = 20;
    int batch_size = 100;
    int num_classes = 10;

    // network
    layer in=Input({784});
    layer l=in;
    l=Reshape(l,{1,28,28});

    l=Block(l,16,{2,2},{1,1});
    l=FMP(l);
    l=Block(l,32,{2,2},{1,1});
    l=Block(l,64,{2,2},{1,1});
    l=Block(l,128,{2,2},{1,1});


    l=Reshape(l,{-1});

    l=Activation(Dense(l,64),"relu");

    layer out=Activation(Dense(l,num_classes),"softmax");

    // net define input and output layers list
    model net=Model({in},{out});

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
            //CS_CPU(4) // 4 CPU threads
            //CS_CPU() // CPU with maximum threads availables
            CS_GPU({1}) // GPU with only one gpu
    );

    // plot the model
    plot(net,"model.pdf");

    // get some info from the network
    summary(net);
  getchar();

    // Load and preprocess training data
    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    eddlT::div_(x_train, 255.0);


    // Load and preprocess test data
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");
    eddlT::div_(x_test, 255.0);

    for(int i=0;i<epochs;i++) {
        // training, list of input and output tensors, batch, epochs
        fit(net,{x_train},{y_train},batch_size, 1);
        // Evaluate train
        std::cout << "Evaluate train:" << std::endl;
        evaluate(net,{x_train},{y_train});
    }


    // Evaluate test
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net,{x_test},{y_test});

}
