/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/generators/layer_generators.h"


using namespace std;

GeneratorLayer::GeneratorLayer(string name, int dev, int mem) : LinLayer(name, dev, mem) {

}


void GeneratorLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void GeneratorLayer::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}
