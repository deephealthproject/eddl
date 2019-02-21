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
#ifndef _LAYER_
#define _LAYER_

#include <string>

#include "tensor.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;

class Layer
{
 public:
  string name;
  Tensor *input;
  Tensor *output;
  Tensor *target;
  Tensor *delta;
  Layer *orig;

  vector<Tensor*>params;
  vector<Tensor*>gradients;

  vector<Layer*> parent;
  vector<Layer*> child;

  int mode;
  int dev;
  int lin,lout;
  int delta_bp;

  Layer(string name);
  Layer(string name,int dev);

  void initialize();
  void reset();

  //virtual
  virtual void info(){};
  virtual void addchild(Layer *l){}
  virtual void addparent(Layer *l){}
  virtual void forward(){}
  virtual void backward(){}
  virtual Layer *share(int c,int bs,vector<Layer*>){return NULL;}
  virtual Layer *clone(int c,int bs,vector<Layer*>,int todev){return NULL;}
};

/////////////////////////////////////////
/////////////////////////////////////////
// Layers with only one input
class LinLayer : public Layer
{
 public:

  LinLayer(string name,int dev);

  void addchild(Layer *l);
  void addparent(Layer *l);

  //virtual

  virtual void info(){};
  virtual void forward(){}
  virtual void backward(){}
  virtual Layer *share(int c,int bs,vector<Layer*> p){return NULL;}
  virtual Layer *clone(int c,int bs,vector<Layer*>,int todev){return NULL;}

};

/// INPUT Layer
class LInput : public LinLayer
{
 public:
  LInput(Tensor *in);
  LInput(Tensor *in,int dev);
  LInput(Tensor *in,string name);
  LInput(Tensor *in,string name,int dev);
  Layer *share(int c,int bs,vector<Layer*>p);
  Layer *clone(int c,int bs,vector<Layer*>,int todev);

  void info();
  void forward();
  void backward();

};

/// DENSE Layer
class LDense : public LinLayer
{
 public:
  int dim;

  LDense(Layer *parent,int dim);
  LDense(Layer *parent,int dim,int dev);
  LDense(Layer *parent,int dim,string name);
  LDense(Layer *parent,int dim,string name,int d);
  Layer *share(int c,int bs,vector<Layer*>p);
  Layer *clone(int c,int bs,vector<Layer*>,int todev);

  // Paras
  Tensor *W;
  Tensor *gW;
  Tensor *bias;
  Tensor *gbias;

  void info();
  void forward();
  void backward();

};

/// DENSE Layer
class LActivation : public LinLayer
{
 public:
  string act;

  LActivation(Layer *parent,string act);
  LActivation(Layer *parent,string act,int d);
  LActivation(Layer *parent,string act,string name);
  LActivation(Layer *parent,string act,string name,int d);
  Layer *share(int c,int bs,vector<Layer*>p);
  Layer *clone(int c,int bs,vector<Layer*>,int todev);

  void info();
  void forward();
  void backward();

};

/////////////////////////////////////////
/////////////////////////////////////////
// Layers with several inputs (ADD, CAT,...)
class MLayer : public Layer
{
 public:

  MLayer(string name,int dev);

  void addchild(Layer *l);
  void addparent(Layer *l);

  //virtual
  virtual void info(){};
  virtual void forward(){}
  virtual void backward(){}
  virtual Layer *share(int c,int bs,vector<Layer*>p){return NULL;}
  virtual Layer *clone(int c,int bs,vector<Layer*>,int todev){return NULL;}
};

/// INPUT Layer
class LAdd : public MLayer
{
 public:
  LAdd(vector<Layer*> in);
  LAdd(vector<Layer*> in,int dev);
  LAdd(vector<Layer*> in,string name);
  LAdd(vector<Layer*> in,string name,int dev);
  Layer *share(int c,int bs,vector<Layer*>p);
  Layer *clone(int c,int bs,vector<Layer*>,int todev);

  void info();
  void forward();
  void backward();

};
#endif
