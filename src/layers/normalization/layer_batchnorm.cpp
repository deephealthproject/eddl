/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_normalization.h"
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

    init=true;
    //
    input=parent->output;

    if (momentum!=0.0) {
        mean=new LTensor(input->getShape(),dev);
        mean->output->fill_(0.0);

        variance=new LTensor(input->getShape(),dev);
        variance->output->fill_(1.0);

        //params.push_back(mean->output);
        //params.push_back(variance->output);
    }

    // create a sub-graph
    LRMean *mean_x,*var;
    LMult *mult;
    LSum *veps;
    LDiff *diff;
    LSqrt *sd;
    LDiv *div;

    // mean
    mean_x=new LRMean(parent, axis, true,this->name+"mean_x",dev);

    // var
    diff=new LDiff(parent, mean_x,this->name+"diff",dev);
    mult=new LMult(diff,diff,this->name+"mult",dev);
    var=new LRMean(mult, axis,true,this->name+"mean_mult",dev);
    //sd
    veps=new LSum(var,epsilon,this->name+"sum_eps",dev);
    sd=new LSqrt(veps,this->name+"sqrt",dev);
    // norm
    div=new LDiv(diff,sd,this->name+"div",dev);

    layers.push_back(mean_x); //0
    layers.push_back(diff);  //1 --
    layers.push_back(mult);  //2
    layers.push_back(var);   //3
    layers.push_back(veps);  //4 --
    layers.push_back(sd);    //5 --
    layers.push_back(div);   //6 --

    // save statistics with momentum
    if (momentum!=0.0) {
      LMult *mx,*mm,*vx,*vm;
      LSum *addm,*addv;

      mx=new LMult(mean_x,(1.0-momentum),this->name+"mx",dev);
      mm=new LMult(mean,momentum,this->name+"mm",dev);
      addm=new LSum(mx,mm,this->name+"sum_m",dev);

      vx=new LMult(var,(1-momentum),this->name+"sx",dev);
      vm=new LMult(variance,momentum,this->name+"sm",dev);
      addv=new LSum(vx,vm,this->name+"sum_sd",dev);

      layers.push_back(mx);  //7
      layers.push_back(mm);  //8
      layers.push_back(addm);//9

      layers.push_back(vx);  //10
      layers.push_back(vm);  //11
      layers.push_back(addv);//12
    }

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

  if (batch==layers[0]->output->shape[0]) return;

  for(int i=0;i<layers.size();i++) layers[i]->resize(batch);

  if (target!=nullptr) target->resize(batch);


  if (momentum!=0.0) {
    if (!init) {
      Tensor *nmean=new Tensor(mean->output->getShape(),dev);
      Tensor *nvar=new Tensor(variance->output->getShape(),dev);

      Tensor::copy(mean->output,nmean);
      Tensor::copy(variance->output,nvar);

      mean->resize(batch);
      variance->resize(batch);

      int msize=mean->output->shape[0];
      int nsize=nmean->shape[0];

      if (msize>nsize) {
        //from nmean to mean with deselect
        vector<int> sind(msize);
        int start,end;
        for(int i=0;i<msize;i++) sind[i]=i;
        for(int i=0;i<msize/nsize;i++) {
            start = i * nsize;
            end = start + nsize;
            Tensor::deselect(nmean, mean->output, sind, start, end);
            Tensor::deselect(nvar, variance->output, sind, start, end);
        }
        if (msize%nsize) {
          Tensor::deselect(nmean, mean->output, sind, end, end+(msize%nsize));
          Tensor::deselect(nvar, variance->output, sind,end, end+(msize%nsize));
        }
      }
      else {
        //from nmean to mean with select
        vector<int> sind(nsize);
        int start,end;
        for(int i=0;i<nsize;i++) sind[i]=i;
        for(int i=0;i<nsize/msize;i++) {
            start = i * msize;
            end = start + msize;
            Tensor::select(nmean, mean->output, sind, start, end);
            Tensor::select(nvar, variance->output, sind, start, end);
        }
        if (nsize%msize) {
          Tensor::select(nmean, mean->output, sind, end, end+(nsize%msize));
          Tensor::select(nvar, variance->output, sind,end, end+(nsize%msize));
        }

      }


      delete nmean;
      delete nvar;
    }
    else {
      mean->resize(batch);
      variance->resize(batch);
      mean->output->fill_(0.0);
      variance->output->fill_(1.0);
    }
  }
}

void LBatchNorm::reset()
{
  for (int i = 0; i != layers.size(); i++)
      layers[i]->reset();
}

void LBatchNorm::forward() {
  init=false;
  if (mode==TRMODE) {
    for(int i=0;i<layers.size();i++) {
      layers[i]->forward();
    }
    if (momentum!=0.0) {
      Tensor::copy(layers[9]->output,mean->output);
      Tensor::copy(layers[12]->output,variance->output);
    }
  }
  else {
    Tensor::copy(mean->output,layers[0]->output);
    layers[1]->forward();

    Tensor::copy(variance->output,layers[3]->output);
    layers[4]->forward();
    layers[5]->forward();
    layers[6]->forward();
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
