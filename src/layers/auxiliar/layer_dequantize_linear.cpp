/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/auxiliar/layer_auxiliar.h"


using namespace std;

int LDequantizeLinear::total_layers = 0;

LDequantizeLinear::LDequantizeLinear(Layer *parent, string name, int dev, int mem, Tensor *x_scale, Tensor *x_zero_point, int axis) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "dequantizelinear" + to_string(++total_layers);
    
    this->size = size;
    this->axis = axis;
    this->x_scale = x_scale;
    this->x_zero_point = x_zero_point;
    input = parent->output;
    cout << this->name << " parent " << parent->name << endl;
    printf("desquantize antes Linear\n");
    if(this->name.compare("input_input_data_int8_dequant")==0){
      printf("PRINT INPUT\n");
      cout << this->x_scale->ptr[0] << " " << this->x_zero_point->ptr[0] << endl;
      this->input->print(); //falla, devuelve solo 1 valor 0, deberia devolver 32 valores
    }
    output = new Tensor(input->shape, dev);
    printf("desquantize despues Linear\n");

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LDequantizeLinear::resize(int batch){
    output->resize(batch);
}


void LDequantizeLinear::forward() {
    printf("FORWARD DEQUANTIZE  CALL FROM LAYER\n");
    tensorNN::dequantize_linear(this->input, this->output, this->x_scale, this->x_zero_point, this->axis);
    if(this->name.compare("input_input_data_int8_dequant")==0){
      printf("PRINT OUTPUT input_input_data_int8_dequant\n");
      this->output->print(); //falla, devuelve solo 1 valor 0, deberia devolver 32 valores
    }
}

void LDequantizeLinear::backward() {
    printf("Error, dequantize_linear layer does not support backward\n"); exit(1);
}


Layer *LDequantizeLinear::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LDequantizeLinear(p[0], "LDequantizeLinear_"+to_string(c)+this->name, this->dev, this->mem_level, this->x_scale, this->x_zero_point, this->axis);
    n->orig = this;

    return n;
}

Layer *LDequantizeLinear::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LDequantizeLinear(p[0], "LDequantizeLinear_" +to_string(c)+this->name, this->dev, this->mem_level, this->x_scale, this->x_zero_point, this->axis);
    n->orig = this;

    return n;
}


string LDequantizeLinear::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
