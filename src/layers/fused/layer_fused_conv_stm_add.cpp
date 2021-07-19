

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/fused/layer_fused.h"

#include "eddl/hardware/fpga/fpga_hw.h"     // FPGA enables of kernels


using namespace std;


int LConvSTMAdd::total_layers = 0;

LConvSTMAdd::LConvSTMAdd(vector<Layer *> parent, int filters, const vector<int> &kernel_size,
                                     const vector<int> &strides, string padding, const vector<int> &pads,
                                     int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem) :
       LConvSTMAdd(parent, new ConvolDescriptor(filters, kernel_size, strides, padding, pads, groups, dilation_rate, use_bias, mem), name, dev, mem) {
};

LConvSTMAdd::LConvSTMAdd(vector<Layer *> parent, ConvolDescriptor *D, string name, int dev, int mem) : MLayer(name, dev, mem){
    if (parent.size() == 0) msg("Error: LConvSTMAdd layer with empty list");
    if (parent.size() != 2)  msg("Error: LConvSTMAdd layer only supports two parent layers");

    if(name.empty()) this->name = "conv2d_stm_add_" + to_string(++total_layers);

    if (parent[0]->output->ndim != 4 || parent[1]->output->ndim != 4) 
        msg("LConvSTMAdd only works over 4D tensors", "LConvSTMAdd::LConvSTMAdd");
    
    input = parent[0]->output;
    cd = D;
    cd->ksize[0] =ceil((float)cd->ksize[0]/CPO) * CPO;
    cd->build(input);

    if (!Tensor::sameShape(cd->O, parent[1]->output)) {
        cd->O->info();
        parent[1]->output->info();
        msg("Error: LConvSTMAdd layers with different tensor shape");
    }

    output = new Tensor(parent[0]->output->shape, dev);

    params.push_back(cd->K);
    params.push_back(cd->bias);

    gradients.push_back(cd->gK);
    gradients.push_back(cd->gbias);

    distributed_training = false;
    cd->acc_gK = nullptr;
    cd->acc_gbias = nullptr;

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}

LConvSTMAdd::~LConvSTMAdd(){
    delete cd;  
}

// virtual
void LConvSTMAdd::resize(int batch){
    cd->resize(batch);
}

// virtual

string LConvSTMAdd::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LConvSTMAdd::forward() { // cambiar
    //we apply the convolutional and stm module to the first layer
    //and then we add the tensor of the second layer
   // printf("\n\n\n PARENT 0\n");
   // _profile_fpga_tensor_print(parent[0]->output);
   // printf("\n\n\n PARENT 1\n");
   // _profile_fpga_tensor_print(parent[1]->output);
    exit(0);
    tensorNN::conv_stm_add(this->cd, parent[0]->output);

}

void LConvSTMAdd::backward() {
    printf("Error, the backward function on fused conv_stm_add layer not supported\n");
    exit(1);
}

Layer *LConvSTMAdd::share(int c, int bs, vector<Layer *> p) {
    LConvSTMAdd *n = new LConvSTMAdd(p,cd->filters, cd->kernel_size, cd->strides, cd->padding, cd->pads, cd->groups, cd->dilation_rate, cd->use_bias, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}


Layer *LConvSTMAdd::clone(int c, int bs, vector<Layer *> p, int todev) {
    LConvSTMAdd *n = new LConvSTMAdd(p,cd->filters, cd->kernel_size, cd->strides, cd->padding, cd->pads, cd->groups, cd->dilation_rate, cd->use_bias, this->name, todev, this->mem_level);

    n->orig = this;

    return n;
}