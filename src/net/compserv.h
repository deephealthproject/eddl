/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_COMPSERV_H
#define EDDL_COMPSERV_H

#include <stdio.h>
#include <string>
#include <vector>

#include <Eigen/Dense>

using namespace std;

class CompServ {
public:
    string type;


    int local_threads;
    vector<int> local_gpus;
    vector<int> local_fpgas;
    int lsb; //local sync batches

    // for local
    CompServ(int threads, const vector<int> &g, const vector<int> &f,int lsb=1);

    // for Distributed
    explicit CompServ(string filename);


};

#endif  //EDDL_COMPSERV_H
