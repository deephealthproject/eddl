/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_NETLOSS_H
#define EDDL_NETLOSS_H

#include <cstdio>
#include <string>
#include <vector>
#include <functional>


#include "eddl/net/net.h"


using namespace std;

class NetLoss {
public:
    string name;
    float value;
    Net* graph;
    vector <Layer *>input;
    vector <Layer *>ginput;
    Layer* fout;

    NetLoss(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    NetLoss(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
    ~NetLoss();
    float compute();

};

#endif  //EDDL_NETLOSS_H
