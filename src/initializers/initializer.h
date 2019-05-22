// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef EDDLL_INITIALIZER_H
#define EDDLL_INITIALIZER_H

#include <string>

#include "../tensor.h"

using namespace std;


class Initializer {
public:
    string name;
    // Todo: Implement
    explicit Initializer(string name);
    virtual float set_weights(Tensor *T);
};

class IConstant : public Initializer {
public:
    float value;

    explicit IConstant(float value);
    float set_weights(Tensor *T) override;
};

class IIdentity : public Initializer {
public:
    float gain;

    explicit IIdentity(float gain);
    float set_weights(Tensor *T) override;
};

class IGlorotNormal : public Initializer {
public:
    int seed;

    explicit IGlorotNormal(int seed=-1);
    float set_weights(Tensor *T) override;
};

class IGlorotUniform : public Initializer {
public:
    int seed;

    explicit IGlorotUniform(int seed=-1);
    float set_weights(Tensor *T) override;
};

class IRandomNormal : public Initializer {
public:
    float mean;
    float stdev;
    int seed;

    explicit IRandomNormal(float mean, float stdev, int seed=-1);
    float set_weights(Tensor *T) override;
};

class IRandomUniform : public Initializer {
public:
    float minval;
    float maxval;
    int seed;

    explicit IRandomUniform(float minval, float maxval, int seed=-1);
    float set_weights(Tensor *T) override;
};

class IOrthogonal : public Initializer {
public:
    float gain;
    int seed;

    explicit IOrthogonal(float gain, int seed=-1);
    float set_weights(Tensor *T) override;
};

#endif //EDDLL_INITIALIZER_H
