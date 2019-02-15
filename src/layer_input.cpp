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
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;

int input_created=1;

Input::Input(Tensor *in):Input(in,"input"+to_string(input_created),DEV_CPU){}
Input::Input(Tensor *in,int d):Input(in,"input"+to_string(input_created),d){}
Input::Input(Tensor *in,string name):Input(in,name,DEV_CPU){}
Input::Input(Tensor *in,string name,int d):LinLayer(name,d)
{
  input_created++;
  input=output=in;
  delta=new Tensor(input->getshape(),d);
}


// virtual
void Input::info()
{
  cout<<"\n===============\n";
  cout<< "Layer Input "<<name<<"\n";
  input->info();
  cout<<"===============\n\n";
}


void Input::forward()
{
  delta->set(0.0);
}


void Input::backward()
{
}


Layer *Input::share(int c,vector<Layer*>p)
{
  shape s=input->getshape();
  s[0]/=c;

  Input *n=new Input(new Tensor(s),"share_"+to_string(c)+name,dev);
  n->orig=this;

  return n;
}

Layer *Input::clone(int c,vector<Layer*>p,int todev)
{
  shape s=input->getshape();
  s[0]/=c;

  Input *n=new Input(new Tensor(s,todev),"clone_"+to_string(todev)+name,todev);
  n->orig=this;

  return n;
}



//////
