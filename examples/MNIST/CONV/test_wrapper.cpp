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

//#include "../../../src/eddl.h"
#include "../../../src/wrapper.h"

layer ResBlock(layer in,int k,int n)
{
    const int ks1_c[] = {k, 1, 1};
    const int ks3_c[] = {k, 3, 3};
    const int st1_c[] = {1, 1};
    const int st2_c[] = {2, 2};

    layer l=in;
    for(int i=0;i<n;i++)

        l=Activation_init(Conv_init(l, ks3_c, 3, st1_c, 2, "same", "Conv"), "relu", "Activation");

    // adapt depth of input
    in=Conv_init(in,ks1_c, 3, st1_c, 2, "same", "Conv");

    // add input and last
    layer l_add[] = {in, l};
    l=Add_init(l_add, 2, "Add");

    // reduce size
    l=Conv_init(l,ks3_c, 3, st2_c, 2, "same", "Conv");
    return l;
}

int main(int argc, char **argv)
{

    // download MNIST data
    //eddl.download_mnist();

    int batch=1000;

    // network
    const int s[] = {batch,784};
    Tensor* t = Tensor_init(s, 2);
    layer in=Input_init(t, "Input");
    layer l=in;

    const int rs_shape[] = {batch,1,28,28};
    l=Reshape_init(l, rs_shape, 4, "Reshape");

    const int ks1_c[] = {16, 3, 3};
    const int ks2_c[] = {32, 3, 3};
    const int ks3_c[] = {64, 3, 3};
    const int ks4_c[] = {128, 3, 3};
    const int st1_c[] = {1,1};
    const int ks2_p[] = {2,2};
    const int st2_p[] = {2,2};
    l=MPool_init(Activation_init(Conv_init(l, ks1_c, 3, st1_c, 2, "same", "Conv"),"relu", "Activation"), ks2_p, 2, st2_p, 2, "none", "MaxPool");
    l=MPool_init(Activation_init(Conv_init(l, ks2_c, 3, st1_c, 2, "same", "Conv"),"relu", "Activation"), ks2_p, 2, st2_p, 2, "none", "MaxPool");
    l=MPool_init(Activation_init(Conv_init(l, ks3_c, 3, st1_c, 2, "same", "Conv"),"relu", "Activation"), ks2_p, 2, st2_p, 2, "none", "MaxPool");
    l=MPool_init(Activation_init(Conv_init(l, ks4_c, 3, st1_c, 2, "same", "Conv"),"relu", "Activation"), ks2_p, 2, st2_p, 2, "none", "MaxPool");

    /*for(int i=0,k=16;i<3;i++,k=k*2)
      l=ResBlock(l,k,2);
  */
    const int shape[] = {batch, -1};
    l=Reshape_init(l, shape, 2, "Reshape");

    l=Activation_init(Dense_init(l, 32, "Dense"), "relu", "Activation");

    layer out=Activation_init(Dense_init(l,10, "Dense"),"softmax", "Activation");

    // net define input and output layers list
    model net=Model_init(in, 1, out, 1);

    // plot the model
    plot(net,"model.pdf");

    // get some info from the network
    summary(net);

    // Attach an optimizer and a list of error criteria and metrics
    // size of error criteria and metrics list must match with size of list of outputs
    // optionally put a DEVICE where the net will run
    optimizer sgd=SGD_init(0.01, 0.9);

    const char* c1 = "soft_cent";
    const char* m1 = "acc";

    const char** c = {&c1};
    const char** m = {&m1};

    compserv cs = CS_CPU_init(4); // local CPU with 6 threads
    //compserv cs=eddl.CS_GPU({1,0,0,0}); // local GPU using the first gpu of 4 installed
    //compserv cs=eddl.CS_GPU({1});// local GPU using the first gpu of 1 installed

    // build(model net, optimizer opt, const char** c, int size_c, const char** m, int size_m, int todev)
    build(net, sgd, c, 1, m, 1, cs);

    // Load and preprocess training data
    tensor X=LTensor_init_fromfile("trX.bin");
    tensor Y=LTensor_init_fromfile("trY.bin");
    LTensor_div(X, 255.0);

    // training, list of input and output tensors, batch, epochs
    fit(net, X->input, Y->input, batch, 1);

    // Evaluate train
    std::cout << "Evaluate train:" << std::endl;
    evaluate(net, X->input, Y->input);

    // Load and preprocess test data
    tensor tX=LTensor_init_fromfile("tsX.bin");
    tensor tY=LTensor_init_fromfile("tsY.bin");
    LTensor_div(tX, 255.0);

    // Evaluate test
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net, tX->input, tY->input);
}


///////////