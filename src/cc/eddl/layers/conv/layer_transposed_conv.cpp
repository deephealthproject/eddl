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

#include "layer_conv.h"


using namespace std;

int LConvT::total_layers = 0;

// ---- TRANSPOSED CONVOLUTION ----
LConvT::LConvT(Layer *parent, int filters, const initializer_list<int> &kernel_size,
    const initializer_list<int> &output_padding, string padding, const initializer_list<int> &dilation_rate,
    const initializer_list<int> &strides, bool use_bias, string name, int dev) : LConvT(parent, new ConvolDescriptor(filters, kernel_size, strides, padding), name, dev) {
    // TODO: Implement (Fix initialization)
};

LConvT::LConvT(Layer *parent, ConvolDescriptor *cd, string name, int dev) : LinLayer(name, dev) {
    // TODO: Implement (Fix initialization)
    if(name.empty()) this->name = "convt" + to_string(++total_layers);
}


