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


using namespace eddl;


//////////////////////////////////
// Text generation
// Only Decoder
//////////////////////////////////


int main(int argc, char **argv) {

    int size = 256/2;

    layer in  = Input({3, 10, size, size});
    layer l=in;
     // Conv3D expects (B,C,dim1,dim2,dim3)
    l=MaxPool3D(ReLu(Conv3D(l,4,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=MaxPool3D(ReLu(Conv3D(l,8,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=MaxPool3D(ReLu(Conv3D(l,16,{1, 3, 3},{1, 1, 1}, "same")),{1, 2, 2}, {1, 2, 2}, "same");
    l=GlobalMaxPool3D(l);
    //l=Squeeze(l);
    l = Reshape(l, {-1});
    l = LSTM(l, 128);
    l = Dense(l, 100);
    l = ReLu(l);
    l = Dense(l, 2);
    layer out = ReLu(l);
    model deepVO = Model({in},{out});

    build(deepVO,
          adam(),
          {"mse"},
          {"mse"},
          CS_GPU({1})
//          CS_CPU()
          );
    plot(deepVO,"model.pdf","TB");
    summary(deepVO);

    // Input: 32 samples that are sequences of 10  3D RGB images of 256x256. 
    Tensor* seqImages = Tensor::randu({32, 10, 3, 10, size, size});
    
    // Target: A sequence of 7 samples of 2 values per image
    Tensor* seqLabels = Tensor::randu({32, 7, 2});


    fit(deepVO, {seqImages}, {seqLabels}, 4, 10);

    return 0;


}
