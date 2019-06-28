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


// AUTOENCODER
int main(int argc, char **argv) {

    // Download dataset
    eddl.download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 1000;

    // Define network
    layer in = eddl.Input({784});
    layer l = in;  // Aux var

    l = eddl.Activation(eddl.Dense(l, 256), "relu");
    l = eddl.Activation(eddl.Dense(l, 128), "relu");
    l = eddl.Activation(eddl.Dense(l, 64), "relu");
    l = eddl.Activation(eddl.Dense(l, 128), "relu");
    l = eddl.Activation(eddl.Dense(l, 256), "relu");
    layer out = eddl.Dense(l, 784);

    model net = eddl.Model({in}, {out});

    // View model
    eddl.summary(net);
    eddl.plot(net, "model.pdf");

    // Build model
    eddl.build(net,
               eddl.sgd(0.01, 0.9), // Optimizer
               {eddl.LossFunc("mean_squared_error")}, // Losses
               {eddl.MetricFunc("mean_squared_error")}, // Metrics
               eddl.CS_CPU(4) // CPU with 4 threads
    );

    // Load dataset
    tensor x_train = eddl.T("trX.bin");
    // Preprocessing
    eddl.div(x_train, 255.0);

    // Train model
    eddl.fit(net, {x_train}, {x_train}, batch_size, epochs);


}


///////////
