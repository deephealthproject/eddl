/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
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

layer ResBlock(layer l, int filters,int nconv,int half) {
  layer in=l;

  if (half)
      l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{2,2})));
  else
      l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{1,1})));


  for(int i=0;i<nconv-1;i++)
    l=ReLu(BatchNormalization(Conv(l,filters,{3,3},{1,1})));

  if (half)
    return Sum(BatchNormalization(Conv(in,filters,{1,1},{2,2})),l);
  else
    return Sum(l,in);
}


Tensor *onehot(Tensor *in, int vocs)
{
  int n=in->shape[0];
  int l=in->shape[1];
  int c=0;

  Tensor *out=new Tensor({n,l,vocs});
  out->fill_(0.0);

  int p=0;
  for(int i=0;i<n*l;i++,p+=vocs) {
    int w=in->ptr[i];
    if (w==0) c++;
    out->ptr[p+w]=1.0;
  }

  cout<<"padding="<<(100.0*c)/(n*l)<<"%"<<endl;
  return out;
}

int main(int argc, char **argv) {

    // ---------------------------
    // Definicion de la red
    // ---------------------------
    layer in  = Input({3, 256, 256});
    layer l=in;

    l=MaxPool(ReLu(Conv(l,2,{3,3},{1,1})),{2,2});
    l=MaxPool(ReLu(Conv(l,4,{3,3},{1,1})),{2,2});
    l=MaxPool(ReLu(Conv(l,8,{3,3},{1,1})),{2,2});
    l=MaxPool(ReLu(Conv(l,16,{3,3},{1,1})),{2,2});
    l=GlobalAveragePool(l);
    l = Flatten(l);
    l = LSTM(l, 128);
    l = Dense(l, 100);
    l = ReLu(l);
    l = Dense(l, 2);
    layer out = ReLu(l);
    model deepVO = Model({in},{out});

    printf("============ OK =============\n");
    // ---------------------------
    // Build
    // ---------------------------
    build(deepVO, adam(), {"mse"}, {"mse"},  CS_CPU() );
    plot(deepVO,"model.pdf","TB");
    summary(deepVO);

    // -------------------------------------------------------------------------------------------------------
    // Fit. Datos de entrenamiento 32 secuencias de 10 imágenes RGB de 256x256
    // ------------------------------------------------------------------------------------------------------
    Tensor* seqImages = Tensor::randu({32, 10, 3, 256, 256});
    Tensor* seqLabels = Tensor::randu({32, 10, 2});
    fit(deepVO, {seqImages}, {seqLabels}, 4, 10);

    return 0;


}
