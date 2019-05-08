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
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "../eddl.h"
#include "../wrapper.h"


float loss(tensor ytrue, tensor ypred) {
    LTensor yt = *ytrue;
    LTensor yp = *ypred;

    LTensor l = yt + yp;


    return 0.0;

}

layer ResBlock(layer in, int dim, int n) {

    layer l = in;
    for (int i = 0; i < n; i++)
        l = Activation_init(Dense_init(l, dim, "Dense"), "relu", "Activation");

    layer l_add[] = {in, l};
    l = Add_init(l_add, 2, "Add");

    return l;
}

int main(int argc, char **argv) {

    int batch = 1000;

    // Download dataset
    eddl.download_mnist();

    // network
    const int s[] = {batch, 784};
    Tensor *t = Tensor_init(s, 2);
    layer in = Input_init(t, "Input");
    layer l = in;
    layer l2;

    l = Drop_init(Activation_init(Dense_init(l, 1024, "Dense"), "relu", "Activation"), 0.5, "drop");
    for (int i = 0; i < 2; i++) {
        if (i == 1) l2 = l;
        l = ResBlock(l, 1024, 1);
    }

    layer l_cat[] = {l, l2};
    l = Cat_init(l_cat, 2, "cat");

    layer out = Activation_init(Dense_init(l, 10, "Dense"), "softmax", "Activation");

    // net define input and output layers list
    model net = Model_init(in, 1, out, 1);

    // plot the model
    plot(net, "model.pdf");

    // get some info from the network
    info(net);

    // Attach an optimizer and a list of error criteria and metrics
    // size of error criteria and metrics list must match with size of list of outputs
    // optionally put a DEVICE where the net will run
    optimizer sgd = SGD_init(0.01, 0.9);

    const char *c1 = "soft_cent";
    const char *m1 = "acc";

    const char **c = {&c1};
    const char **m = {&m1};

    compserv cs = CS_CPU_init(4); // local CPU with 6 threads

    // build(model net, optimizer opt, const char** c, int size_c, const char** m, int size_m, int todev)
    build(net, sgd, c, 1, m, 1, cs);


    // Load and preprocess training data
    tensor X = LTensor_init_fromfile("trX.bin");
    tensor Y = LTensor_init_fromfile("trY.bin");
    LTensor_div(X, 255.0);

    // training, list of input and output tensors, batch, epochs
    fit(net, X->input, Y->input, batch, 1);

    // Evaluate train
    std::cout << "Evaluate train:" << std::endl;
    evaluate(net, X->input, Y->input);

    // Evaluate test
    tensor tX = LTensor_init_fromfile("tsX.bin");
    tensor tY = LTensor_init_fromfile("tsY.bin");
    LTensor_div(tX, 255.0);

    std::cout << "Evaluate test:" << std::endl;
    evaluate(net, tX->input, tY->input);

}


///////////
