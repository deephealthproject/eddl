/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include "eddl/net/net.h"
#include <pthread.h>
#include "eddl/utils.h"
#include "eddl/random.h"
#include "eddl/layers/core/layer_core.h"

#define VERBOSE 0

using namespace std;
using namespace std::chrono;

/////////////////////////////////////////////////////////////////
///// NET LEVEL FUNCS
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////
void Net::do_initialize() {
  for (int i = 0; i != layers.size(); i++)
  layers[i]->initialize();
}

/////////////////////////////////////////
void Net::do_reset() {
  for (int i = 0; i != layers.size(); i++) {
    layers[i]->reset();
  }
}

void Net::do_reset_grads() {
  for (int i = 0; i != layers.size(); i++) {
    layers[i]->zeroGrads();
  }
}

void Net::do_forward() {
  if (VERBOSE) {
    cout<<"START FORWARD\n";
    getchar();
  }
  for (int i = 0; i < vfts.size(); i++) {
    if (VERBOSE) {
      cout << vfts[i]->name << " Shape: ";
      vfts[i]->info();
      for(int j=0;j<vfts[i]->parent.size();j++)
      fprintf(stdout, "  %s In[%d,%s]:%f\n", vfts[i]->name.c_str(), j, vfts[i]->parent[j]->name.c_str(),vfts[i]->parent[j]->output->sum());
    }

    vfts[i]->forward();
    if (VERBOSE) {
      fprintf(stdout, "  %s Out:%f\n", vfts[i]->name.c_str(), vfts[i]->output->sum());
      getchar();
    }
  }
  if (VERBOSE) {
    cout<<"END FORWARD\n";
    getchar();
  }
}

void Net::do_backward() {
  if (VERBOSE) {
    cout<<"START BACKWARD\n";
    getchar();
  }
  for (int i = 0; i < vbts.size(); i++) {
    if(this->verbosity_level >= 1){
      std::cout << vbts[i]->name << std::endl;
    }

    // Reserve parent's delta (if reserved, ignored)
    vbts[i]->mem_delta_parent();

    // Do backward
    if (VERBOSE) {
      cout << "backward "<<vbts[i]->name << " delta="<<vbts[i]->delta->sum()<<"\n";
    }

    vbts[i]->backward();


    // Delete this delta
    if(vbts[i]->mem_level) { vbts[i]->free_delta(); }
  }
  if (VERBOSE) {
    cout<<"END BACKWARD\n";
    getchar();
  }
}

void Net::do_delta() {
  if (VERBOSE) {
    cout<<"Delta\n";
    getchar();
  }
  for (int i = 0; i < lout.size(); i++) {
    lout[i]->mem_delta();
    if (losses.size()>=(i+1)) {
      losses[i]->delta(lout[i]->target, lout[i]->output, lout[i]->delta);
      if (VERBOSE) cout<<"Delta: "<<lout[i]->name<<" delta:"<<lout[i]->delta->sum()<<"\n";
    }
  }
  if (VERBOSE) {
    cout<<"Delta end\n";
    getchar();
  }
}

void Net::do_compute_loss() {
  if (VERBOSE) {
    cout<<"Compute Loss\n";
    getchar();
  }

  int p = 0;
  for (int i = 0; i < lout.size(); i++, p += 2) {
    // loss value
    if (losses.size()>=(i+1))
    fiterr[p] = losses[i]->value(lout[i]->target, lout[i]->output);
    // metric value
    if (metrics.size()>=(i+1))
    fiterr[p + 1] = metrics[i]->value(lout[i]->target, lout[i]->output);
  }

  if (VERBOSE) {
    cout<<"Compute Loss end\n";
    getchar();
  }
}

void Net::do_applygrads() {
  optimizer->applygrads(batch_size);
}




void Net::sync_weights() {
  //cout<<"\nSync weights...\n";
  for (int j = 0; j < layers.size(); j++)
  for (int k = 0; k < layers[j]->params.size(); k++) {
    // Taking average
    layers[j]->params[k]->fill_(0.0);
    for (int i = 0; i < snets.size(); i++) {
      Tensor::inc(snets[i]->layers[j]->params[k], layers[j]->params[k]);
    }
    layers[j]->params[k]->div_(snets.size());

    // copy-back to devices
    for (int i = 0; i < snets.size(); i++) {
      Tensor::copy(layers[j]->params[k], snets[i]->layers[j]->params[k]);
    }

  }
}


void collectTensor(Layer *l,string tname, int p)
{
  Net *sn=l->net;
  if (sn->snets[0]->dev==DEV_CPU) return;

  int i,j,comp;

  comp=sn->snets.size();

  if ((l->output->ndim==1)&&(comp>1)) {
    cout<<"Warning "<<l->name<<" samples lower than Computing Service\n";
    cout<<"Normally it means that you have used some reduction layer that avoids data parallelism\n";
    comp=1;
  }
  else if (l->output->shape[0]<comp) {
    comp=l->output->shape[0];
  }

  int thread_batch_size=l->output->shape[0] / comp;

  vector<int> sind(l->output->shape[0]);
  for(int k=0;k<l->output->shape[0];k++) sind[k]=k;

  for(i=0;i<comp;i++) {
    Layer *sl=nullptr;

    for(j=0;j<sn->snets[i]->layers.size();j++) {
      if (sn->snets[i]->layers[j]->orig==l) {
        sl=sn->snets[i]->layers[j];
        break;
      }
    }
    if (sl==nullptr) {
      cout<<"LAYER:"<<l->name<<"\n";
      msg("layer not found in subgrap","collectTensor");
    }

    int start = i * thread_batch_size;
    int end = start + sl->output->shape[0];

    if (tname=="output")
    Tensor::deselect(sl->output, l->output, sind, start, end);
    else if (tname=="delta")
    Tensor::deselect(sl->delta, l->delta, sind, start, end);
    else if (tname=="param")
    Tensor::copy(sl->params[p],l->params[p]);
    else if (tname=="gradient")
    Tensor::copy(sl->gradients[p],l->gradients[p]);
  }
}


void distributeTensor(Layer *l,string tname, int p)
{
  Net *sn=l->net;

  if (sn->snets[0]->dev==DEV_CPU) return;

  int i,j,comp;

  comp=sn->snets.size();

  if (sn->batch_size<comp) {
    msg("batch_size lower than computing service parallelism","distributeTensor");

  }
  int thread_batch_size=sn->batch_size / comp;

  vector<int> sind(sn->batch_size);
  for(int k=0;k<sn->batch_size;k++) sind[k]=k;


  for(i=0;i<sn->snets.size();i++) {
    Layer *sl=nullptr;

    for(j=0;j<sn->snets[i]->layers.size();j++)
    if (sn->snets[i]->layers[j]->orig==l) {
      sl=sn->snets[i]->layers[j];
      break;
    }

    if (sl==nullptr) {
      cout<<l->name<<"\n";
      msg("layer not found in subgrap","distributeTensor");
    }

    int start = i * thread_batch_size;
    int end = start + sl->output->shape[0];

    if (tname=="output")
    Tensor::select(l->output, sl->output, sind, start, end);
    else if (tname=="delta") {
      sl->mem_delta();
      Tensor::select(l->delta, sl->delta, sind, start, end);
    }
    else if (tname=="param")
    Tensor::copy(l->params[p],sl->params[p]);
    else if (tname=="gradient")
    Tensor::copy(l->gradients[p],sl->gradients[p]);
  }
}
