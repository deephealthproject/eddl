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

#include "../tensor.h"

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
  void info();


  //virtual
  virtual string plot(int c){return "";}

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


  virtual string plot(int c){return "";}
  virtual void forward(){}
  virtual void backward(){}
  virtual Layer *share(int c,int bs,vector<Layer*> p){return NULL;}
  virtual Layer *clone(int c,int bs,vector<Layer*>,int todev){return NULL;}

};

/// INPUT Layer
class LTensor : public LinLayer
{
 public:
  LTensor(string fname);
  LTensor(const initializer_list<int>& init);
  LTensor(const initializer_list<int>& init, int dev);
  LTensor(const shape s);
  LTensor(const shape s, int dev);

  Layer *share(int c,int bs,vector<Layer*>p){return NULL;}
  Layer *clone(int c,int bs,vector<Layer*>,int todev){return NULL;}
  void info(){}
  void forward(){}
  void backward(){}
  string plot(int c){return "";}

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

  void forward();
  void backward();
  string plot(int c);

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

  void forward();
  void backward();
  string plot(int c);

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

  void forward();
  void backward();
  string plot(int c);

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

  virtual string plot(int c){return "";}
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

  void forward();
  void backward();
  string plot(int c);

};




class LReshape : public LinLayer
{
 public:
  shape ls;

  // constructors and clones
  LReshape(Layer *parent,const initializer_list<int>& init);
  LReshape(Layer *parent,const initializer_list<int>& init,int dev);
  LReshape(Layer *parent,const initializer_list<int>& init,string name);
  LReshape(Layer *parent,const initializer_list<int>& init,string name,int d);
  LReshape(Layer *parent,shape s);
  LReshape(Layer *parent,shape s,int d);
  LReshape(Layer *parent,shape s,string name);
  LReshape(Layer *parent,shape s,string name,int d);

  Layer *share(int c,int bs,vector<Layer*>p);
  Layer *clone(int c,int bs,vector<Layer*>p,int todev);

  // Params

  // implementation
  void forward();
  void backward();
  string plot(int c);

};


class LDrop : public LinLayer
{
 public:
  int dim;

  // constructors and clones
  LDrop(Layer *parent,float df);
  LDrop(Layer *parent,float df,int dev);
  LDrop(Layer *parent,float df,string name);
  LDrop(Layer *parent,float df,string name,int d);
  Layer *share(int c,int bs,vector<Layer*>p);
  Layer *clone(int c,int bs,vector<Layer*>p,int todev);

  float df;
  Tensor *mask;

  // implementation
  void forward();
  void backward();
  string plot(int c);

};


class LCat : public MLayer
{
 public:
  int dim;
  vector<int> index;

  // constructors and clones
  LCat(vector<Layer*> in);
  LCat(vector<Layer*> in,int dev);
  LCat(vector<Layer*> in,string name);
  LCat(vector<Layer*> in,string name,int d);
  Layer *share(int c,int bs,vector<Layer*>p);
  Layer *clone(int c,int bs,vector<Layer*>p,int todev);

  // Params


  // implementation
  void forward();
  void backward();
  string plot(int c);

};
#endif
