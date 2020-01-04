/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_reductions.h"


using namespace std;

ReductionLayer::ReductionLayer(string name, int dev) : Layer(name, dev) {
    binary=0;
}


void ReductionLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void ReductionLayer::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}

string ReductionLayer::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
