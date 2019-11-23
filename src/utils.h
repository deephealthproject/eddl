/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_UTILS_H
#define EDDL_UTILS_H

#include <string>
#include <cstdio>
#include <cstdint> // uint64_t

using namespace std;


float *get_fmem(int size,char *str);

char *humanSize(uint64_t bytes);

unsigned long get_free_mem();

string get_extension(string filename);

#endif //EDDL_UTILS_H
