/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "layers/generators/layer_generators.h"


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
