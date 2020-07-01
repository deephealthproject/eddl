/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/reductions/layer_reductions.h"


using namespace std;

ReductionLayer2::ReductionLayer2(string name, int dev, int mem) : Layer(name, dev, mem) {
    binary=0;
}

void ReductionLayer2::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void ReductionLayer2::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}

string ReductionLayer2::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
