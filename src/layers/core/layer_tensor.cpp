
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_core.h"
#include "../merge/layer_merge.h"  // TODO: Review dependency (LADD)


using namespace std;


int LTensor::total_layers = 0;


// From file
LTensor::LTensor(string fname) : LinLayer("ltensor" + to_string(total_layers), DEV_CPU) {
    Tensor* t = new Tensor();
    t->load(fname);
    data = input = output = t;
}

LTensor::LTensor(const vector<int> shape, float *fptr,int dev) : LinLayer("ltensor" + to_string(total_layers), dev) {
    data = input = output = new Tensor(shape, fptr, dev);
    delta = new Tensor(shape, dev);
}

// From vector<int>
LTensor::LTensor(const vector<int> shape, int dev) : LinLayer("ltensor" + to_string(total_layers), dev) {
    data = input = output = new Tensor(shape, dev);
    delta = new Tensor(shape, dev);
}


/*
void Ltensor::mult2D(LTensor *A,...){
  Tensor::mult2d(A->output, tA,B->output,tB,C->output,incC);
}
*/
void LTensor::resize(int batch){
  Layer::resize(batch);
}


// From Layer
LTensor::LTensor(Layer *l) : LinLayer("ltensor" + to_string(total_layers), l->dev) {
    data = input = output = l->output;
    delta = l->delta;
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
