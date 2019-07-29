
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

#ifndef EDDLL_UTILS_H
#define EDDLL_UTILS_H

float uniform(float min=0.0f, float max=1.0f);

float suniform();

float gaussgen();

float gauss(float mean, float sd);

void gen_rtable();

float gauss(int s, float mean, float sd);

float *get_fmem(int size,char *str);

char *humanSize(uint64_t bytes);

#endif //EDDLL_UTILS_H
