/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_TENSOR_RED_H
#define EDDL_TENSOR_RED_H

#include "tensor.h"
#include "../descriptors/descriptors.h"

using namespace std;


void reduction(ReduceDescriptor *RD);
void reduction_back(ReduceDescriptor *RD);

#endif
