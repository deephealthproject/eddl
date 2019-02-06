// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019 Roberto Paredes Palacios, <rparedes@dsic.upv.es>

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

#include "../net.h"

int main(int argc, char **argv)
{

  int batch=100;

  Tensor *tin=new Tensor({batch,784});

  // network definition
  Input* in=new Input(tin);
  Layer *l=in;
  for(int i=0;i<3;i++)
    l=new Activation(new Dense(l,512),"relu");

  Activation *out=new Activation(new Dense(l,10),"softmax");

  // define input and output layers list
  Net *net=new Net({in},{out});

  // get some info from the network
  net->info();

  // Attach an optimizer and a list of error criteria
  // size of error criteria list must match with size of list of outputs
  net->build(SGD(0.001,0.9),{"soft_cent"});

  /// read data somewhere
  Tensor *X=new Tensor("trX.bin");
  Tensor *Y=new Tensor("trY.bin");

  X->div(255.0);

  // training, list of input and output tensors, batch, epochs
  net->fit({X},{Y},batch,1);


}


















  ///////////
