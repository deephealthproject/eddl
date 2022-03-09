/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/operators/layer_operators.h"


using namespace std;

OperatorLayer::OperatorLayer(string name, int dev, int mem) : Layer(name, dev, mem) {
    binary=0;
}


void OperatorLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void OperatorLayer::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}

string OperatorLayer::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=diamond]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=Beige,shape=diamond]";

    return s;
}
