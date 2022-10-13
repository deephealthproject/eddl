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

#include "eddl/layers/merge/layer_merge.h"
#include "eddl/layers/operators/layer_operators.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/profiling.h"
#include "eddl/hardware/fpga/fpga_hw.h"

using namespace std;

int LConcat::total_layers = 0;

LConcat::LConcat(vector<Layer *> parent, unsigned int axis, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if(name.empty()) {this->name = "concat" + to_string(++total_layers); }

    // Perform layer checks
    if (parent.empty()) { msg("Error: LConcat layer with empty list"); }

    // Check special cases
    if(axis==-1){
        // -1 because last index is shape-1, and -1 (again) to remove the batch component
        axis = (parent[0]->output->ndim - 1) - 1;
    }else if(axis<0){
        msg("The axist must be greater or equal than zero", "LConcat::LConcat");
    }

    this->axis = axis; // batch is not included

    if (parent.size() > 1) {

        // All layers need to have the same number of dimensions
        int ndim = parent[0]->output->ndim;
        for (int i = 0; i < parent.size() - 1; ++i) {
            if (ndim != parent[i + 1]->output->ndim){
                msg("Error: LConcat layers with different tensor dims");
            }
        }
        // Check dimensions (10, 5, 3, 3) + (10, 2, 3, 3) => (10, 7, 3, 3)
        for (int i = 0; i < parent.size() - 1; ++i){
            for (int d = 0; d < ndim; ++d) {
                if (d != (this->axis+1) && parent[i]->output->shape[d] != parent[i + 1]->output->shape[d]) {
                    msg("Error: LConcat layers with different size in dim 1 (" +
                    to_string(parent[i]->output->shape[d])  + "!=" +
                    to_string(parent[i + 1]->output->shape[d]) + ")");
                }
            }
        }

    }else{
        msg("Error: LConcat must receive at least two layers");
    }

    // Sum depth dimensions to know the final dimension [2+2+6=10]
    int t = 0;
    for (int i = 0; i < parent.size(); ++i) {
        t += parent[i]->output->shape[this->axis+1];
    }

    // Set new shape
    vector<int> new_shape = parent[0]->output->getShape();
    new_shape[this->axis+1] = t;

    input = parent[0]->output;
    output = new Tensor(new_shape, dev);

    // Set children
    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}

#ifdef cFPGA
void LConcat::allocate_output_fpga_buffer() {
   if (output->fpga_ptr == NULL) {
    if (hlsinf_output_format == HLSINF_FP32) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*sizeof(float));
    } else if (hlsinf_output_format == HLSINF_API8) {
      output->fpga_ptr = fpga_create_memory(output->size);
    } else if (hlsinf_output_format == HLSINF_APF_8_4) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
    } else if (hlsinf_output_format == HLSINF_APF_16_8) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
    } else if (hlsinf_output_format == HLSINF_APF_32_16) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*sizeof(ap_fixed<32,16>));
    } else if (hlsinf_output_format == HLSINF_APUI8) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size);
    } else {
      printf("Error (HLSinf forward), output format not supported\n");
      exit(1);
    }
  }
}
#endif


void LConcat::forward() {

#ifdef cFPGA
    if (layer_disabled) return;
#endif

    // Get output tensors
    vector<Tensor*> outputs;
    for (auto & p : this->parent) { outputs.push_back(p->output); }

    // Perform concatenation
    Tensor::concat(outputs, this->axis+1, this->output);
}


void LConcat::backward() {
    // Get delta tensors
    vector<Tensor*> deltas;
    for (int i=0; i<this->parent.size(); i++) {
        // Store pointer of delta i
        deltas.push_back(parent[i]->delta);
    }

    // Perform concat (back)
    Tensor::concat_back(this->delta, deltas, this->axis+1);
}

Layer *LConcat::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LConcat(p, this->axis, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LConcat::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LConcat(p, this->axis, name, todev,mem_level);
    n->orig = this;

    return n;
}


string LConcat::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}
