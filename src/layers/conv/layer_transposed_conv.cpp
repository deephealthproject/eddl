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

#include "layer_conv.h"


using namespace std;

int LConvT::total_layers = 0;

// ---- TRANSPOSED CONVOLUTION ----
LConvT::LConvT(Layer *parent, int filters, const vector<int> &kernel_size,
    const vector<int> &output_padding, string padding, const vector<int> &dilation_rate,
    const vector<int> &strides, bool use_bias, string name, int dev) : LConvT(parent, new ConvolDescriptor(filters, kernel_size, strides, padding), name, dev) {
    // TODO: Implement (Fix initialization)
};

LConvT::LConvT(Layer *parent, ConvolDescriptor *cd, string name, int dev) : LinLayer(name, dev) {
    // TODO: Implement (Fix initialization)
    if(name.empty()) this->name = "convt" + to_string(++total_layers);
}
