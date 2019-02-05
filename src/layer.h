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

#ifndef _LAYER_
#define _LAYER_

#include <string>

#include "tensor.h"


#define TRMODE 1
#define TSMODE 0

using namespace std;

class Layer {
 public:
  string name;
  Tensor *input;
  Tensor *output;
  Tensor *delta;

  vector<Tensor*>params;
  vector<Tensor*>gradients;


  vector<Layer*> parent;
  vector<Layer*> child;

  int mode;
  int dev;
  int lin,lout;

  Layer(string name);
  Layer(string name,int dev);

  void initialize();
  void reset();
  void applygrads();

  //virtual
  virtual void info(){};
  virtual void addchild(Layer *l){}
  virtual void addparent(Layer *l){}
  virtual void forward(){}
  virtual void backward(){}
};


/////////////////////////////////////////
/////////////////////////////////////////
// Layers with only one input
class LinLayer : public Layer {
 public:

  LinLayer(string name,int dev);

  void addchild(Layer *l);
  void addparent(Layer *l);

  //virtual
  virtual void info(){};
  virtual void forward(){}
  virtual void backward(){}

};


/// INPUT Layer
class Input : public LinLayer {
 public:
  Input(Tensor *in);
  Input(Tensor *in,int dev);
  Input(Tensor *in,string name);
  Input(Tensor *in,string name,int dev);

  void info();
  void forward();
  void backward();

};

/// DENSE Layer
class Dense : public LinLayer {
 public:
  int dim;

  Dense(Layer *parent,int dim);
  Dense(Layer *parent,int dim,int dev);
  Dense(Layer *parent,int dim,string name);
  Dense(Layer *parent,int dim,string name,int d);
  // Paras
  Tensor *W;
  Tensor *gW;
  Tensor *bias;
  Tensor *gbias;



  void info();
  void forward();
  void backward();

};



/////////////////////////////////////////
/////////////////////////////////////////
// Layers with several inputs (ADD, CAT,...)
class MLayer : public Layer {
 public:



  MLayer(string name,int dev);

  void addchild(Layer *l);
  void addparent(Layer *l);

  //virtual
  virtual void info(){};
  virtual void forward(){}
  virtual void backward(){}

};

/// INPUT Layer
class Add : public MLayer {
 public:
  Add(vector<Layer*> in);
  Add(vector<Layer*> in,int dev);
  Add(vector<Layer*> in,string name);
  Add(vector<Layer*> in,string name,int dev);

  void info();
  void forward();
  void backward();

};

#endif
