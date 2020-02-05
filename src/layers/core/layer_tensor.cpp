/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include "layer_core.h"
#include "../merge/layer_merge.h"  // TODO: Review dependency (LADD)


using namespace std;


int LTensor::total_layers = 0;

extern ostream &operator<<(ostream &os, const vector<int> shape);

// From shape
LTensor::LTensor(const vector<int> shape, int dev, int mem) : LinLayer("ltensor" + to_string(total_layers++), dev) {
    data = input = output = new Tensor(shape, dev);
    if (!mem_level) { delta = new Tensor(output->shape, dev); }
}


LTensor::~LTensor()
{
  input = output = nullptr;
}

// From file
LTensor::LTensor(string fname) : LinLayer("ltensor" + to_string(total_layers), DEV_CPU) {
  Tensor *t = Tensor::load(fname, "bin");
  data = input = output = t;
}

// From file
LTensor * LTensor::fromCSV(string fname) {
  FILE *fe = fopen(fname.c_str(), "rt");
  if (fe == nullptr) {
      throw std::runtime_error(fname + " not found");
  }
  vector<int> shape;
  int ndim,v;

  fscanf(fe,"%d",&ndim);
  for(int i=0;i<ndim;i++) {
    fscanf(fe,"%d",&v);
    shape.push_back(v);
  }

  LTensor *n=new LTensor(shape,DEV_CPU);

  for (int i = 0; i < n->output->size; ++i)
      fscanf(fe,"%f ",&(n->output->ptr[i]));

  return n;

}


// From shape, ptr (sharing) and device
LTensor::LTensor(const vector<int> shape, float *fptr,int dev, int mem) : LinLayer("ltensor" + to_string(total_layers), dev) {
    data = input = output = new Tensor(shape, fptr, dev);
    if (!mem_level) { delta = new Tensor(output->shape, dev); }
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
