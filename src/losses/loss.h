
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

#ifndef _LOSS_
#define _LOSS_

#include <stdio.h>
#include <string>

#include "../tensor/tensor.h"
#include "../tensor/nn/tensor_nn.h"

using namespace std;

class Loss {
public:
    string name;

    explicit Loss(string name);

    virtual void delta(Tensor *T, Tensor *Y, Tensor *D);
    virtual float value(Tensor *T, Tensor *Y);
};


class LMeanSquaredError : public Loss {
public:
    LMeanSquaredError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};


class LCrossEntropy : public Loss {
public:
    LCrossEntropy();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};


class LSoftCrossEntropy : public Loss {
public:
    LSoftCrossEntropy();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};


// TODO: Implement
//void mean_squared_error(Tensor *T, Tensor *Y, Tensor *D);
//void mean_absolute_error(Tensor *T, Tensor *Y, Tensor *D);
//void mean_absolute_percentage_error(Tensor *T, Tensor *Y, Tensor *D);
//void mean_squared_logarithmic_error(Tensor *T, Tensor *Y, Tensor *D);
//void squared_hinge(Tensor *T, Tensor *Y, Tensor *D);
//void hinge(Tensor *T, Tensor *Y, Tensor *D);
//void categorical_hinge(Tensor *T, Tensor *Y, Tensor *D);
//void logcosh(Tensor *T, Tensor *Y, Tensor *D);
//void categorical_crossentropy(Tensor *T, Tensor *Y, Tensor *D);
//void sparse_categorical_crossentropy(Tensor *T, Tensor *Y, Tensor *D);
//void binary_crossentropy(Tensor *T, Tensor *Y, Tensor *D);
//void kullback_leibler_divergence(Tensor *T, Tensor *Y, Tensor *D);
//void poisson(Tensor *T, Tensor *Y, Tensor *D);
//void cosine_proximity(Tensor *T, Tensor *Y, Tensor *D);

#endif
