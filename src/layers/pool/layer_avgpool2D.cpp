/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/pool/layer_pool.h"


using namespace std;


// ---- MAXPOOL2D ----
// constructors and clones

// constructors and clones
LAveragePool::LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem) : LAveragePool(parent, new PoolDescriptor(pool_size, strides, padding, mem), name, dev, mem) {}

LAveragePool::LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, string name, int dev, int mem) : LAveragePool(parent, new PoolDescriptor(pool_size, strides, padding, mem), name, dev, mem) {}

LAveragePool::LAveragePool(Layer *parent, PoolDescriptor *D, const string& name, int dev, int mem) : LPool(parent, D, name, dev, mem) {
    if(name.empty()) this->name = "avgpool" + to_string(++total_layers);
 
    // Params
    D->indX = new Tensor(D->O->shape, dev);  // Is this needed here?
    D->indY = new Tensor(D->O->shape, dev);
#ifdef cCUDNN
if(!D->I->isCPU()){
    D->mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    D->maxpoolingNanOpt = CUDNN_NOT_PROPAGATE_NAN;

    cudnnStatus_t bbb =  cudnnSetPooling2dDescriptor(D->poolingDesc, D->mode, D->maxpoolingNanOpt, D->windowHeight, D->windowWidth,
    D->verticalPadding, D->horizontalPadding, D->verticalStride, D->horizontalStride);
   if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"Error create avg pooling 2D descriptor "<< cudnnGetErrorString(bbb) <<std::endl;
}
#endif

    // Check padding asymmetries
    if(D->pad[0] != D->pad[1] || D->pad[2] != D->pad[3]){
        msg("Padding asymmetry detected. (top=" + to_string(D->pad[0]) + ", bottom=" + to_string(D->pad[1]) + ", left=" + to_string(D->pad[2]) + ", right=" + to_string(D->pad[3]) + ").\nLayer name: " + this->name, "LAveragePool::LAveragePool");
    }
}


void LAveragePool::resize(int batch){
    LPool::resize(batch);

    delete pd->indX; 
    pd->indX = new Tensor(pd->O->shape, dev);
    
    delete pd->indY; 
    pd->indY = new Tensor(pd->O->shape, dev);
}

void LAveragePool::forward() {
    tensorNN::AvgPool2D(this->pd);
}

void LAveragePool::backward() {
    tensorNN::AvgPool2D_back(this->pd);
}

Layer *LAveragePool::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LAveragePool(p[0], new PoolDescriptor(pd->ksize, pd->stride, pd->pad, pd->mem_level), "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LAveragePool::clone(int c, int bs, vector<Layer *> p, int todev) {

    auto *n = new LAveragePool(p[0], new PoolDescriptor(pd->ksize, pd->stride, pd->pad, pd->mem_level),  "share_"+to_string(c)+this->name, todev, this->mem_level);

    n->orig = this;

    return n;
}

string LAveragePool::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
