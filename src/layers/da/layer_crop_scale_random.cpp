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
#include <utility>

#include "layer_da.h"


using namespace std;

int LCropAndScaleRandom::total_layers = 0;

LCropAndScaleRandom::LCropAndScaleRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, string da_mode, string name, int dev) : LCropRandom(parent, std::move(factor_x), std::move(factor_y), 0.0f, name, dev) {
    if(name.empty()) this->name = "crop_scale" + to_string(++total_layers);
    this->da_mode=da_mode;
}

void LCropAndScaleRandom::forward() {
    Tensor::crop_scale_random(this->input, this->output, this->factor_x, this->factor_y, this->da_mode, this->constant);
}
