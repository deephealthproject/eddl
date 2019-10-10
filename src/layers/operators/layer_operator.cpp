/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_operators.h"


using namespace std;

OperatorLayer::OperatorLayer(string name, int dev) : Layer(name, dev) {
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
