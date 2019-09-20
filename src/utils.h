
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdint> // uint64_t

#ifndef EDDL_UTILS_H
#define EDDL_UTILS_H


float *get_fmem(int size,char *str);

char *humanSize(uint64_t bytes);

unsigned long get_free_mem();

#endif //EDDL_UTILS_H
