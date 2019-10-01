
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
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "cpu_nn.h"


void cpu_cent(Tensor *A, Tensor *B, Tensor *C){
    for (int i = 0; i < A->size; i++) {
        C->ptr[i] = 0;
        if (A->ptr[i] != 0.0) C->ptr[i] -= A->ptr[i] * std::log(B->ptr[i]+0.00001);
        if (A->ptr[i] != 1.0) C->ptr[i] -= (1.0 - A->ptr[i]) * std::log(1.0 - B->ptr[i]+0.00001);
    }
}