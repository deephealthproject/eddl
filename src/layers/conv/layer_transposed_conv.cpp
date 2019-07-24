
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

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


