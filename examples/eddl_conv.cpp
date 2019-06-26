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

#include "apis/eddl.h"

int main(int argc, char **argv)
{

  // download MNIST data
  eddl.download_mnist();

  // Settings
  int epochs = 5;
  int num_classes = 10;

  // network
  layer in=eddl.Input({784});
  layer l=in;


  l=eddl.Reshape(l,{1,28,28});
  l=eddl.MaxPool(eddl.Activation(eddl.Conv(l, 16, {3,3}),"relu"),{2,2});
  l=eddl.MaxPool(eddl.Activation(eddl.Conv(l, 32, {3,3}),"relu"),{2,2});
  l=eddl.MaxPool(eddl.Activation(eddl.Conv(l, 64, {3,3}),"relu"),{2,2});
  l=eddl.MaxPool(eddl.Activation(eddl.Conv(l, 128, {3,3}),"relu"),{2,2});

  /*for(int i=0,k=16;i<3;i++,k=k*2)
    l=ResBlock(l,k,2);
*/
  l=eddl.Reshape(l,{-1});

  l=eddl.Activation(eddl.Dense(l,32),"relu");

  layer out=eddl.Activation(eddl.Dense(l,num_classes),"softmax");

  // net define input and output layers list
  model net=eddl.Model({in},{out});

  // plot the model
  eddl.plot(net,"model.pdf");

  // get some info from the network
  cout << eddl.summary(net) << endl;

  // Attach an optimizer and a list of error criteria and metrics
  // optionally put a Computing Service where the net will run
  // size of error criteria and metrics list must match with size of list of outputs
  optimizer sgd=eddl.sgd(0.01,0.9);

  compserv cs=eddl.CS_CPU(6); // local CPU with 4 threads
  //compserv cs=eddl.CS_GPU({1,0,0,0}); // local GPU using the first gpu of 4 installed
  //compserv cs=eddl.CS_GPU({1});// local GPU using the first gpu of 1 installed

  eddl.build(net, sgd, {eddl.LossFunc("soft_cross_entropy")}, {eddl.MetricFunc("categorical_accuracy")}, cs);

  // Load and preprocess training data
  tensor X=eddl.T("trX.bin");
  tensor Y=eddl.T("trY.bin");
  eddl.div(X,255.0);

  // training, list of input and output tensors, batch, epochs
  int batch_size = 1000;
  eddl.fit(net,{X},{Y},batch_size, epochs);

  // Evaluate train
  std::cout << "Evaluate train:" << std::endl;
  eddl.evaluate(net,{X},{Y});

  // Load and preprocess test data
  tensor tX=eddl.T("tsX.bin");
  tensor tY=eddl.T("tsY.bin");
  eddl.div(tX,255.0);

  // Evaluate test
  std::cout << "Evaluate test:" << std::endl;
  eddl.evaluate(net,{tX},{tY});

}


///////////
