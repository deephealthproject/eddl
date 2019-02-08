// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
// 	     Roberto Paredes Palacios, <rparedes@dsic.upv.es>
// 	     Jon Ander Gómez, <jon@dsic.upv.es>
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
#ifndef _NET_
#define _NET_

#include <string>
#include <initializer_list>
#include <vector>

#include "layer.h"
#include "optim.h"
#include "loss.h"
#include "metric.h"

using namespace std;

typedef vector<Layer*> vlayer;
typedef vector<Tensor*> vtensor;
typedef vector<string> vstring;
typedef vector<float> verr;
typedef vector<int> vind;
typedef vector<Loss*> vloss;
typedef vector<Metric*> vmetrics;

class Net {
 public:
  string name;

  vlayer layers;
  vlayer lin;
  vlayer lout;
  vlayer vfts;
  vlayer vbts;


  vind ind;
  vind sind;
  vloss losses;
  vmetrics metrics;
  verr fiterr;
  vstring strcosts;
  vstring strmetrics;
  vector<Net *> snets;
  
  optim *optimizer;


  Net(const initializer_list<Layer*>& in,const initializer_list<Layer*>& out);
  Net(vlayer in,vlayer out);

  int inNet(Layer *);
  void walk(Layer *l);
  Layer *getLayer(string name);

  void build(optim *opt,const initializer_list<string>& c,const initializer_list<string>& m);
  void build(optim *opt,vstring in,vstring out);

  void initialize();
  void reset();
  void forward();
  void delta(vtensor out);
  void loss(vtensor out);
  void backward();
  void applygrads(int batch);
  void info();
  void split(int c);

  void fts();
  void bts();
  void fit(const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out,int batch,int epochs);

  void train_batch(const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out);
  void train_batch(vtensor X,vtensor Y);


};


#endif
