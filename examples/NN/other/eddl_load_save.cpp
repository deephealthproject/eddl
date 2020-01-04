/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace eddl;


int main(int argc, char **argv) {

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = BatchNormalization(ReLu(Dense(l, 1024)));
    l = BatchNormalization(ReLu(Dense(l, 1024)));
    l = BatchNormalization(ReLu(Dense(l, 1024)));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_CPU() // CPU with maximum threads availables
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


    // Save model before the training
    cout << "Saving untrained model..." << endl;
    save(net,"model_untrained.bin");

    // Train model
    cout << "Training model..." << endl;
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Save model after the training
    cout << "Saving trained model..." << endl;
    save(net,"model_trained.bin");

    // Evaluate model before the training
    cout << "Loading untrained model..." << endl;
    load(net,"model_untrained.bin");
    cout << "Evaluating untrained model..." << endl;
    evaluate(net, {x_test}, {y_test});

    // Evaluate model after the training
    cout << "Loading trained model..." << endl;
    load(net,"model_trained.bin");
    cout << "Evaluating trained model..." << endl;
    evaluate(net, {x_test}, {y_test});



}


///////////
