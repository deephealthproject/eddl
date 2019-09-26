
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



#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

void cpu_conv2D(ConvolDescriptor *D);
void cpu_conv2D_grad(ConvolDescriptor *D);
void cpu_conv2D_back(ConvolDescriptor *D);

void cpu_mpool2D(PoolDescriptor*D);
void cpu_mpool2D_back(PoolDescriptor *D);
