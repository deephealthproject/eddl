/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_core.h"
#include "../merge/layer_merge.h"  // TODO: Review dependency (LADD)


using namespace std;


int LTensor::total_layers = 0;

extern ostream &operator<<(ostream &os, const vector<int> shape);

// From shape
LTensor::LTensor(const vector<int> shape, int dev) : LinLayer("ltensor" + to_string(total_layers++), dev) {
    data = input = output = new Tensor(shape, dev);
    delta = new Tensor(shape, dev);
}


// From file
LTensor::LTensor(string fname) : LinLayer("ltensor" + to_string(total_layers), DEV_CPU) {
  FILE *fe = fopen(fname.c_str(), "rb");
  if (fe == nullptr) {
      fprintf(stderr, "%s not found\n", fname.c_str());
      exit(1);
  }

  vector<int> shape;
  int ndim,v;
  int read = fread(&ndim, sizeof(int), 1, fe);
  for (int i = 0; i < ndim; ++i) {
      int read = fread(&v, sizeof(int), 1, fe);
      shape.push_back(v);
  }

  Tensor *t=new Tensor(shape,DEV_CPU);
  cout << "loading file with tensor:" << shape << "\n";
  t->load(fe);
  data = input = output = t;
}

// From shape, ptr (sharing) and device
LTensor::LTensor(const vector<int> shape, float *fptr,int dev) : LinLayer("ltensor" + to_string(total_layers), dev) {
    data = input = output = new Tensor(shape, fptr, dev);
    delta = new Tensor(shape, dev);
}



Layer *LTensor::share(int c, int bs, vector<Layer *> p) {

    LTensor *n = new LTensor(output->shape, dev);
    n->orig = this;

    return n;
}

Layer *LTensor::clone(int c, int bs, vector<Layer *> p, int todev) {

    LTensor *n = new LTensor(output->shape, todev);
    n->orig = this;

    return n;
}


// From Layer (sharing)
LTensor::LTensor(Layer *l) : LinLayer("ltensor" + to_string(total_layers), l->dev) {
    data = input = output = l->output;
    delta = l->delta;
}

void LTensor::resize(int batch){
  Layer::resize(batch);
}



/// OP OVERLOAD
LTensor LTensor::operator+(LTensor L) {
    vector<Layer *> vl;

    vl.push_back(this);
    vl.push_back(&L);

    LTensor *l = new LTensor(new LAdd(vl, "", DEV_CPU));

    return *l;
}





//////
