/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdint> // uint64_t

#ifndef EDDL_UTILS_H
#define EDDL_UTILS_H


float *get_fmem(int size,char *str);

char *humanSize(uint64_t bytes);

unsigned long get_free_mem();

#endif //EDDL_UTILS_H
