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

layer ResBlock(layer in, int dim,int n)
{
  layer l=in;
  for(int i=0;i<n;i++)
    l=eddl.Activation(eddl.Dense(l,dim),"relu");

  l=eddl.Add({in,l});

  return l;

}

int main(int argc, char **argv)
{

  int batch=1000;

  // network
  layer in=eddl.Input({batch,784});
  layer l=in;
  layer l2;

  l=eddl.Drop(eddl.Activation(eddl.Dense(l,1024),"relu"),0.5);
  for(int i=0;i<2;i++) {
      if (i==1) l2=l;
      l=ResBlock(l,1024,1);
  }


  //l=eddl.Reshape(l,{batch,16,2,2,-1});
  //l=eddl.Reshape(l,{batch,1024});

  l=eddl.Cat({l,l2});

  layer out=eddl.Activation(eddl.Dense(l,10),"softmax");

  // net define input and output layers list
  model net=eddl.Model({in},{out});

  // plot the model
  eddl.plot(net,"model.pdf");

  // get some info from the network
  eddl.info(net);

  // Attach an optimizer and a list of error criteria and metrics
  // size of error criteria and metrics list must match with size of list of outputs
  // optionally put a DEVICE where the net will run
  eddl.build(net,SGD(0.01,0.9),{"soft_cent"},{"acc"},DEV_CPU);

  // read data
  tensor X=eddl.T("trX.bin");
  tensor Y=eddl.T("trY.bin");

  eddl.div(X,255.0);

  // training, list of input and output tensors, batch, epochs
  eddl.fit(net,{X},{Y},batch,100);

}


///////////
