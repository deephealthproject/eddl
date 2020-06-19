/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/merge/layer_merge.h"  // TODO: Review dependency (LADD)


using namespace std;


int LTensor::total_layers = 0;

extern ostream &operator<<(ostream &os, const vector<int> shape);

// From shape
LTensor::LTensor(const vector<int> shape, int dev, int mem) : LinLayer("ltensor" + to_string(total_layers++), dev, mem) {
    data = input = output = new Tensor(shape, dev);
//    if (!mem_level) { delta = new Tensor(output->shape, dev); }
}


LTensor::~LTensor()
{
  input = output = nullptr;
}

// From file
LTensor::LTensor(string fname) : LinLayer("ltensor" + to_string(total_layers), DEV_CPU, 0) {
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

  LTensor *n=new LTensor(shape, DEV_CPU, 0);

  for (int i = 0; i < n->output->size; ++i)
      fscanf(fe,"%f ",&(n->output->ptr[i]));

  return n;

}


// From shape, ptr (sharing) and device
LTensor::LTensor(const vector<int> shape, float *fptr,int dev, int mem) : LinLayer("ltensor" + to_string(total_layers), dev, mem) {
    data = input = output = new Tensor(shape, fptr, dev);
    if (!mem_level) { delta = new Tensor(output->shape, dev); }
}



Layer *LTensor::share(int c, int bs, vector<Layer *> p) {

    LTensor *n = new LTensor(output->shape, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LTensor::clone(int c, int bs, vector<Layer *> p, int todev) {

    LTensor *n = new LTensor(output->shape, todev, this->mem_level);
    n->orig = this;

    return n;
}


// From Layer (sharing)
LTensor::LTensor(Layer *l) : LinLayer("ltensor" + to_string(total_layers), l->dev, l->mem_level) {
    data = input = output = l->output;
    delta = l->delta;
}


/// OP OVERLOAD
LTensor LTensor::operator+(LTensor L) {
    vector<Layer *> vl;

    vl.push_back(this);
    vl.push_back(&L);

    LTensor *l = new LTensor(new LAdd(vl, "", DEV_CPU, 0));

    return *l;
}





//////
