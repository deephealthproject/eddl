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

#include "layer_da.h"


using namespace std;

int LCropAndScale::total_layers = 0;

LCropAndScale::LCropAndScale(Layer *parent, vector<int> from_coords, vector<int> to_coords, float constant, string name, int dev) : LCrop(parent, from_coords, to_coords, false, constant, name, dev) {
    if(name.empty()) this->name = "crop_scale" + to_string(++total_layers);
}

void LCropAndScale::forward() {
    Tensor::crop_scale(this->input, this->output, this->from_coords, this->to_coords, this->constant);
}
