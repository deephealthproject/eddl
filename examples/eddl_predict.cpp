// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "eddl.h"
#include "eddl.h"

int main(int argc, char **argv) {

    // Download dataset
    eddl.download_mnist();

    // Settings


    int num_classes = 10;

    // Define network
    layer in = eddl.Input({784});
    layer l = in;  // Aux var

    l = eddl.Activation(eddl.Dense(l, 1024), "relu");
    l = eddl.Activation(eddl.Dense(l, 1024), "relu");
    l = eddl.Activation(eddl.Dense(l, 1024), "relu");
    layer out = eddl.Activation(eddl.Dense(l, num_classes), "softmax");
    model net = eddl.Model({in}, {out});

    // View model
    eddl.summary(net);
    eddl.plot(net, "model.pdf");

    // Build model
    eddl.build(net,
               eddl.sgd(0.01, 0.9), // Optimizer
               {eddl.LossFunc("soft_cross_entropy")}, // Losses
               {eddl.MetricFunc("categorical_accuracy")}, // Metrics
               eddl.CS_CPU(6) // CPU with 4 threads
    );

    // Load dataset
    tensor x_train = eddl.T("trX.bin");
    tensor y_train = eddl.T("trY.bin");
    tensor x_test = eddl.T("tsX.bin");
    tensor y_test = eddl.T("tsY.bin");

    // Preprocessing
    eddl.div(x_train, 255.0);
    eddl.div(x_test, 255.0);

    // Train model
    int batch_size = 1000;
    int epochs = 1;
    eddl.fit(net, {x_train}, {y_train}, batch_size, epochs);

    ///  Predict *one* sample
    float *X=(float*)malloc(1*784*sizeof(float));
    tensor TX=eddl.T({1,784},X);

    float *Y=(float*)malloc(1*10*sizeof(float));
    tensor TY=eddl.T({1,10},Y);

    eddl.predict(net,{TX},{TY});

    // The result is in float *Y
    // but in general you can get the pointer to
    // tensor data by:
    float *result=eddl.T_getptr(TY);


}


///////////
