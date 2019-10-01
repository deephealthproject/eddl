
#ifndef EDDL_TENSOR_RED_H
#define EDDL_TENSOR_RED_H

#include "tensor.h"
#include "../descriptors/descriptors.h"

using namespace std;


void reduction(ReduceDescriptor *RD);
void reduction_back(ReduceDescriptor *RD);

#endif
