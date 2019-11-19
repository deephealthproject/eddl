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
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include "net.h"
#include <pthread.h>
#include "../utils.h"
#include "../random.h"
#include "../layers/core/layer_core.h"

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
    for (int i = 0; i != layers.size(); i++)
        layers[i]->zeroGrads();
}

void Net::do_forward() {
    for (int i = 0; i < vfts.size(); i++) {
        vfts[i]->forward();
        if (VERBOSE) {
          cout << vfts[i]->name << "\n";
          fprintf(stdout, "  %s In:%f\n", vfts[i]->name.c_str(), vfts[i]->input->sum());
          fprintf(stdout, "  %s Out:%f\n", vfts[i]->name.c_str(), vfts[i]->output->sum());
        }
    }
}

void Net::do_backward() {
    for (int i = 0; i < vbts.size(); i++) {
        vbts[i]->backward();
        if (VERBOSE) cout<<"BACK: "<<vbts[i]->name<<"delta:"<<vbts[i]->delta->sum()<<"\n";
      }
    if (VERBOSE) getchar();
}

void Net::do_delta() {
    for (int i = 0; i < lout.size(); i++) {
        losses[i]->delta(lout[i]->target, lout[i]->output, lout[i]->delta);
        if (VERBOSE) cout<<"Delta: "<<vbts[i]->name<<"delta:"<<vbts[i]->delta->sum()<<"\n";
      }
    if (VERBOSE) getchar();
}

void Net::do_compute_loss() {
    int p = 0;
    for (int i = 0; i < lout.size(); i++, p += 2) {
        // loss value
        fiterr[p] = losses[i]->value(lout[i]->target, lout[i]->output);
        // metric value
        fiterr[p + 1] = metrics[i]->value(lout[i]->target, lout[i]->output);
    }
}

void Net::do_applygrads() {
    optimizer->applygrads(batch_size);
}


/////////////////////////////////////////
void Net::sync_weights() {

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


void collectTensor(Layer *l,string tname)
{
  Net *sn=l->net;
  if (sn->snets[0]->dev==DEV_CPU) return;

  int i,j,comp;

  comp=sn->snets.size();

  if (sn->batch_size<comp)
    comp=sn->batch_size;

  int thread_batch_size=sn->batch_size / comp;

  vector<int> sind(sn->batch_size);
  for(int k=0;k<sn->batch_size;k++) sind[k]=k;

  for(i=0;i<sn->snets.size();i++) {
    Layer *sl=nullptr;

    for(j=0;j<sn->snets[i]->layers.size();j++) {
     if (sn->snets[i]->layers[j]->orig==l) {
         sl=sn->snets[i]->layers[j];
         break;
     }
   }
    if (sl==nullptr) {
      cout<<"LAYER:"<<l->name<<"\n";
      msg("layer not found in subgrap","Net.collectTensor");
    }

    int start = i * thread_batch_size;
    int end = start + sl->output->shape[0];

    if (tname=="output")
      Tensor::deselect(sl->output, l->output, sind, start, end);
    else if (tname=="grad")
      Tensor::deselect(sl->delta, l->delta, sind, start, end);
  }

}


void distributeTensor(Layer *l,string tname)
{
  Net *sn=l->net;
  if (sn->snets[0]->dev==DEV_CPU) return;

  int i,j,comp;

  comp=sn->snets.size();

  if (sn->batch_size<comp)
    comp=sn->batch_size;

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
      msg("layer not found in subgrap","Net.distributeTensor");
    }

    int start = i * thread_batch_size;
    int end = start + sl->output->shape[0];

    if (tname=="output")
      Tensor::select(l->output, sl->output, sind, start, end);
    else
      Tensor::select(l->delta, sl->delta, sind, start, end);

  }
}


void Net::copyTensor(Layer *l1,Layer *l2,string name){

  Layer *sl1;
  Layer *sl2;

  if (snets[0]->dev==DEV_CPU) {
    sl1=l1;
    sl2=l2;
  }
  else {
    for(int i=0;i<snets.size();i++) {
      for(int j=0;j<snets[i]->layers.size();j++) {
        if (snets[i]->layers[j]->orig==l1)
        sl1=snets[i]->layers[j];
      }
      for(int j=0;j<snets[i]->layers.size();j++) {
        if (snets[i]->layers[j]->orig==l2)
        sl2=snets[i]->layers[j];
      }
    }
  }

  if (name=="output") Tensor::copy(sl1->output,sl2->output);
  else if (name=="grad") Tensor::copy(sl1->delta,sl2->delta);
}








//////
