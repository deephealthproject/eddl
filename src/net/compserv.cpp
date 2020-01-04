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
#include <string>
#include <vector>

#include "compserv.h"


// for local
CompServ::CompServ(int t, const vector<int> &g, const vector<int> &f,int lsb) {
    type = "local";

    if (t==-1) local_threads = Eigen::nbThreads();
    else local_threads = t;

    local_gpus = vector<int>(g.begin(), g.end());
    local_fpgas = vector<int>(f.begin(), f.end());

    this->lsb=lsb;
    if (lsb<0) {
      fprintf(stderr,"Error creating CS with lsb<0 in CompServ::CompServ");
      exit(EXIT_FAILURE);
    }
}

// for Distributed
CompServ::CompServ(string filename) {
     //TODO: Implement
}
