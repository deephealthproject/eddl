
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

    vector<int> axis;
    if (parent->output->ndim == 2) axis.push_back(0);
    else if (parent->output->ndim == 4) {axis.push_back(0);axis.push_back(2);axis.push_back(3);}
    else msg("LBatchNorm only works over 2D or 4D tensors", "LBatchNorm");

    if(name.empty()) this->name = "batchnorm" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    //
    input=parent->output;

    if (momentum!=0.0) {
      mean=new LTensor(input->getShape(),dev);
      mean->output->set(0.0);

      variance=new LTensor(input->getShape(),dev);
      variance->output->set(1.0);
    }



    // create a sub-graph
    LRMean *mean_x,*var;
    LMult *mx,*mm,*vx,*vm,*mult;
    LSum *addm,*addv,*veps;
    LDiff *diff;
    LSqrt *sd;
    LDiv *div;

    mean_x=new LRMean(parent, axis, true,this->name+"mean_x",dev);
    // momentum to mean
    if (momentum!=0.0) {
      mx=new LMult(mean_x,(1.0-momentum),this->name+"mx",dev);
      mm=new LMult(mean,momentum,this->name+"mm",dev);
      addm=new LSum(mx,mm,this->name+"sum_m",dev);
    }
    diff=new LDiff(parent, mean_x,this->name+"diff",dev);

    // obtain var
    mult=new LMult(diff,diff,this->name+"mult",dev);
    var=new LRMean(mult, axis,true,this->name+"mean_mult",dev);
    //
    // momentum to var
    if (momentum!=0.0) {
      vx=new LMult(var,(1-momentum),this->name+"sx",dev);
      vm=new LMult(variance,momentum,this->name+"sm",dev);
      addv=new LSum(vx,vm,this->name+"sum_sd",dev);
    }

    veps=new LSum(var,epsilon,this->name+"sum_eps",dev);
    sd=new LSqrt(veps,this->name+"sqrt",dev);
    div=new LDiv(diff,sd,this->name+"div",dev);

    layers.push_back(mean_x); //0
    if (momentum!=0.0) {
      layers.push_back(mx);  //1
      layers.push_back(mm);  //2
      layers.push_back(addm);//3
    }

    layers.push_back(diff);  //4 --
    layers.push_back(mult);  //5
    layers.push_back(var);   //6

    if (momentum!=0.0) {
      layers.push_back(vx);  //7
      layers.push_back(vm);  //8
      layers.push_back(addv);//9
    }
    layers.push_back(veps);  //10
    layers.push_back(sd);    //11 --
    layers.push_back(div);   //12 --

    // detach from the main graph
    parent->detach(mean_x);
    parent->detach(diff);
    ////////////////////////////

    output=div->output;
    delta=div->delta;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LBatchNorm::resize(int batch){

  if ((momentum!=0.0)&&(batch==mean->output->shape[0])) return;

  for(int i=0;i<layers.size();i++) layers[i]->resize(batch);

  if (target!=nullptr) target->resize(batch);

  if (momentum!=0.0) {
    mean->resize(batch);
    mean->output->set(0.0);

    variance->resize(batch);
    variance->output->set(1.0);
  }
}

void LBatchNorm::reset()
{
  for (int i = 0; i != layers.size(); i++)
      layers[i]->reset();
}

void LBatchNorm::forward() {
  if (mode==TRMODE) {
    for(int i=0;i<layers.size();i++) {
      layers[i]->forward();
    }
    if (momentum!=0.0) {
      Tensor::copy(layers[9]->output,variance->output);
      Tensor::copy(layers[3]->output,mean->output);
    }
  }
  else {
    Tensor::copy(mean->output,layers[0]->output);
    Tensor::copy(variance->output,layers[10]->output);
    layers[4]->forward();
    layers[11]->forward();
    layers[12]->forward();
  }

}

void LBatchNorm::backward() {
  for(int i=layers.size()-1;i>=0;i--) {
    layers[i]->backward();
  }
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
