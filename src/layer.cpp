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

#include "layer.h"

using namespace std;

////////////////////////////////////
///// BASE LAYER CLASS
////////////////////////////////////
Layer::Layer()
{
  mode=TRMODE;
  delta=input=output=NULL;
}

Layer::Layer(string n):Layer(n,DEV_CPU,"FLOAT32"){}
Layer::Layer(string n,int d):Layer(n,d,"FLOAT32"){}
Layer::Layer(string n,string t):Layer(n,DEV_CPU,t){}

Layer::Layer(string n,int d,string t):Layer()
{
  dev=d;
  type=t;
  name=n;
  if ((t=="FLOAT32")||(t=="FLOAT64")||(t=="INT32")) type=t;
  else msg("Layer unkown type",t);
}


void Layer::initialize()
{
  for(int i = 0; i != vparams.size(); i++)
    vparams[i]->rand();
}

void Layer::reset()
{

}


////////////////////////////////////
///// LINEAR LAYERS
////////////////////////////////////
LinLayer::LinLayer(string n,int d,string t):Layer(n,d,t)
{
    parent=NULL;
    child=NULL;
}

void LinLayer::addchild(Layer *l)
{
  if (child!=NULL) msg("LinLayers only can have one child layer\nTry with Split Layer");
  child=l;
}
void LinLayer::addparent(Layer *l)
{
    if (parent!=NULL) msg("LinLayers only can have one parent layer");
    parent=l;
}








//////
