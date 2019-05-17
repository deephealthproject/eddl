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

#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "initializer.h"

using namespace std;

Initializer::Initializer(string name) {
    this->name = name;
}
float Initializer::set_weights(Tensor *T) {
    return 0;
}

IConstant::IConstant(float value) : Initializer("constant") {
    // Todo: Implement
    this->value = value;
}
float IConstant::set_weights(Tensor *T){}

IIdentity::IIdentity(float gain) : Initializer("identity") {
    // Todo: Implement
    this->gain = gain;
}
float IIdentity::set_weights(Tensor *T){}

IGlorotNormal::IGlorotNormal(int seed) : Initializer("glorot_normal") {
    // Todo: Implement
    this->seed = seed;
}
float IGlorotNormal::set_weights(Tensor *T){}

IGlorotUniform::IGlorotUniform(int seed) : Initializer("glorot_uniform") {
    // Todo: Implement
    this->seed = seed;
}
float IGlorotUniform::set_weights(Tensor *T){}

IRandomNormal::IRandomNormal(float mean, float stdev, int seed) : Initializer("random_normal") {
    // Todo: Implement
    this->mean = mean;
    this->stdev = stdev;
    this->seed = seed;

}
float IRandomNormal::set_weights(Tensor *T){}

IRandomUniform::IRandomUniform(float minval, float maxval, int seed) : Initializer("random_uniform") {
    // Todo: Implement
    this->minval = minval;
    this->maxval = maxval;
    this->seed = seed;

}
float IRandomUniform::set_weights(Tensor *T){}

IOrthogonal::IOrthogonal(float gain, int seed) : Initializer("orthogonal") {
    // Todo: Implement
    this->gain = gain;
    this->seed = seed;

}
float IOrthogonal::set_weights(Tensor *T){}