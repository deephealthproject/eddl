
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

int cpu_accuracy(Tensor *A, Tensor *B){
    int acc = 0;
    int aind, bind;

    for (int i = 0; i < A->shape[0]; i++) {
        (*A->ptr2).col(i).maxCoeff(&aind);
        (*B->ptr2).col(i).maxCoeff(&bind);
        if (aind == bind) acc++;
    }
    return acc;
}
