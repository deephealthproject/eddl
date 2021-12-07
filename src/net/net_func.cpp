/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "eddl/net/net.h"
#include "eddl/utils.h"
#include "eddl/random.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/mpi_distributed/mpi_distributed.h"


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

    for (int i = 0; i < vfts.size(); i++)
        vfts[i]->forward();

}

void Net::do_backward() {
    for (int i = 0; i < vbts.size(); i++) {
        //if (!vbts[i]->trainable) return;

        vbts[i]->mem_delta_parent();

        vbts[i]->backward();

        if(vbts[i]->mem_level) { vbts[i]->free_delta(); }
    }
}

void Net::do_delta() {
    for (int i = 0; i < lout.size(); i++) {
        lout[i]->mem_delta();
        if (losses.size()>=(i+1)) {
            losses[i]->delta(lout[i]->target, lout[i]->output, lout[i]->delta);
        }
    }
}

void Net::do_compute_loss() {
    int p = 0;
    for (int i = 0; i < lout.size(); i++, p += 2) {
        // loss value
        if (losses.size()>=(i+1)){
            fiterr[p] = losses[i]->value(lout[i]->target, lout[i]->output);
        }
        // metric value
        if (this->metrics.size()>=(i+1)){
            fiterr[p + 1] = this->metrics[i]->value(lout[i]->target, lout[i]->output);
        }
    }
}

void Net::do_applygrads() {
    optimizer->applygrads(batch_size);
}

void Net::collect_acc_grads() {
    for (int j = 0; j < layers.size(); j++)
        for (int k = 0; k < layers[j]->acc_gradients.size(); k++) {
            // Taking average
            layers[j]->acc_gradients[k]->fill_(0.0);
            for (int i = 0; i < snets.size(); i++)
                Tensor::inc(snets[i]->layers[j]->acc_gradients[k], layers[j]->acc_gradients[k]);
            layers[j]->acc_gradients[k]->div_(snets.size());
        }
}

void Net::distribute_weights() {
    msg("Not implemented error", "Net::distribute_weights");
}

void Net::sync_weights() {
        printf("==== sync_weights ====\n");
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


void collectTensor(Layer *l,string tname, int p){
    // This function collects the tensors from the snets (parallel networks in specific devices), copying them
    // to the net object (always in CPU), that acts as a coordinator.
    // e.g: snet #1 (GPU; 50% batch): 50% outputs
    //      snet #2 (GPU; 50% batch): 50% outputs
    //      collectTensor(net, "output") => net (CPU) => 100% outputs

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
            msg("layer not found in subgrap","Net::collectTensor");
        }

        int start = i * thread_batch_size;
        int end = start + sl->output->shape[0];

        if (tname=="output"){
            Tensor::deselect(sl->output, l->output, sind, start, end);
        }else if (tname=="input"){
            Tensor::deselect(sl->input, l->input, sind, start, end);
        }else if (tname=="delta"){
            Tensor::deselect(sl->delta, l->delta, sind, start, end);
        }else if (tname=="param"){
            Tensor::copy(sl->params[p],l->params[p]);
        }else if (tname=="gradient"){
            Tensor::copy(sl->gradients[p],l->gradients[p]);
        }else if (tname=="state"){
            Tensor::copy(sl->states[p],l->states[p]);
        }else {
            msg("Unknown name (" + tname + ")","Net::collectTensor");
        }

    }
}


void distributeTensor(Layer *l,string tname, int p)
{
    Net *sn=l->net;

    if (sn->snets[0]->dev==DEV_CPU) return;

    int i,j,comp;
    vector<int> sind(sn->batch_size);
    int thread_batch_size;

    if ((tname=="output")||(tname=="delta")) {
        // output or deltas
        // check batch_size and comp
        comp=sn->snets.size();
        if (sn->batch_size<comp) {
            msg("batch_size lower than computing service parallelism","distributeTensor");
        }
        thread_batch_size=sn->batch_size / comp;
        for(int k=0;k<sn->batch_size;k++) sind[k]=k;
    }


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

        if (tname=="output") {
            Tensor::select(l->output, sl->output, sind, start, end);
        }
        else if (tname=="delta") {
            sl->mem_delta();
            Tensor::select(l->delta, sl->delta, sind, start, end);
        }
        else if (tname=="param") {
            cout<<"Distribute param "<<p<<" to device "<<i<<endl;
            Tensor::copy(l->params[p],sl->params[p]);
        }
        else if (tname=="gradient") {
            Tensor::copy(l->gradients[p],sl->gradients[p]);
        }
    }

}

