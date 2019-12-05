/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_NETLOSS_H
#define EDDL_NETLOSS_H

#include <stdio.h>
#include <string>
#include <vector>
#include <functional>


#include "net.h"


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
