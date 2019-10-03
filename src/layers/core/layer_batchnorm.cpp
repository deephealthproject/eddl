
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
#include "../reductions/layer_reductions.h"
#include "../operators/layer_operators.h"

using namespace std;

int LBatchNorm::total_layers = 0;

LBatchNorm::LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev) : LinLayer(name, dev) {

    if (parent->output->ndim != 2) msg("LBatchNorm only works over 2D tensors", "LBatchNorm");
    if(name.empty()) this->name = "batchnorm" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    ///
    input=parent->output;
    output=parent->output;
    delta=parent->delta;

    mean=new LTensor(output->getShape(),dev);
    mean->output->set(0.0);

    sd=new LTensor(output->getShape(),dev);
    sd->output->set(1.0);

    vector<int> axis;
    axis.push_back(0);

    // create a sub-graph
    LRMean *mean_x=new LRMean(this, axis, true,this->name+"mean_x",dev);
    // momentum to mean
    LMult *mx=new LMult(mean_x,(1.0-momentum),this->name+"mx",dev);
    LMult *mm=new LMult(mean,momentum,this->name+"mm",dev);
    LSum *addm=new LSum(mx,mm,this->name+"sum_m",dev);

    // obtain sd
    LDiff *diff=new LDiff(this, addm,this->name+"diff",dev);
    LMult *mult=new LMult(diff,diff,this->name+"mult",dev);
    LRMean *var=new LRMean(mult, axis,true,this->name+"var",dev);
    LSum *vareps=new LSum(var,epsilon,this->name+"vareps",dev);

    // momentum to sd
    LMult *sx=new LMult(vareps,(1-momentum),this->name+"sx",dev);
    LMult *sm=new LMult(sd,momentum,this->name+"sm",dev);
    LSum *adds=new LSum(sx,sm,this->name+"sum_sd",dev);

    LSqrt *sd_x=new LSqrt(adds,this->name+"sd_x",dev);

    LDiv *div=new LDiv(diff, adds,this->name+"divs",dev);

    layers.push_back(mean_x);
    layers.push_back(mx);
    layers.push_back(mm);
    layers.push_back(addm);

    layers.push_back(diff);
    layers.push_back(mult);
    layers.push_back(var);
    layers.push_back(vareps);


    layers.push_back(sx);
    layers.push_back(sm);
    layers.push_back(adds);
    layers.push_back(sd_x);
    layers.push_back(div);

    // detach from the main graph
    detach(mean_x);
    detach(diff);
    ////////////////////////////

    output=div->output;
    delta=div->delta;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LBatchNorm::resize(int batch){
  for(int i=0;i<layers.size();i++) layers[i]->resize(batch);

  if (target!=nullptr) target->resize(batch);

  mean->resize(batch);
  mean->output->set(0.0);

  sd->resize(batch);
  sd->output->set(1.0);
}

void LBatchNorm::forward() {
  for(int i=0;i<layers.size();i++) {
    layers[i]->forward();
  }
}

void LBatchNorm::backward() {
  for(int i=layers.size()-1;i>=0;i--) layers[i]->backward();
}


Layer *LBatchNorm::share(int c, int bs, vector<Layer *> p) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LBatchNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LBatchNorm::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
