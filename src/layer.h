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


#include <string>
#include "tensor.h"

#define MAX_CONNECT 100
#define TRMODE 0
#define TSMODE 1

using namespace std;

class Layer {
 public:
  string name;
  Tensor *input;
  Tensor *output;
  Tensor *delta;


  vector<Tensor*>vparams;
  vector<Tensor*>gradients;

  int mode;
  int dev;
  string type;

  Layer();
  Layer(string name);
  Layer(string name,string type);
  Layer(string name,int dev);
  Layer(string name,int dev,string type);

  void initialize();
  void reset();

  //virtual
  virtual void info(){};
  virtual void addchild(Layer *l){}
  virtual void addparent(Layer *l){}
  virtual void forward(){}
  virtual void backward(){}
  virtual void applygrads(){}
};

class LinLayer : public Layer {
 public:
  Layer *parent;
  Layer *child;

  LinLayer(string name,int dev,string type);

  void addchild(Layer *l);
  void addparent(Layer *l);

  //virtual
  virtual void info(){};
  virtual void forward(){}
  virtual void backward(){}
  virtual void applygrads(){}

};

/// INPUT Layer
class Input : public LinLayer {
 public:

  Input(Tensor *in,string name);

  void info();
  void forward();
  void backward();
  void applygrads();

};

/// DENSE Layer
class Dense : public LinLayer {
 public:
  int dim;


  Dense(Layer *parent,int dim,string name);
  Dense(Layer *parent,int dim,string name,int d);
  Dense(Layer *parent,int dim,string name,string t);
  Dense(Layer *parent,int dim,string name,int d,string t);
  // Paras
  Tensor *W;
  Tensor *gW;
  Tensor *bias;
  Tensor *gbias;



  void info();
  void forward();
  void backward();
  void applygrads();

};
