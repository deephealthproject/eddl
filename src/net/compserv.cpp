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
#include <stdexcept>

#include "compserv.h"


// for local
CompServ::CompServ(int t, const vector<int> g, const vector<int> &f,int lsb, int mem) {
    type = "local";

    if (t==-1) local_threads = Eigen::nbThreads(); // TODO: Review => std::thread::hardware_concurrency()???
    else local_threads = t;

    local_gpus = vector<int>(g.begin(), g.end());
    local_fpgas = vector<int>(f.begin(), f.end());

    this->lsb=lsb;

    if (lsb<0) {
      throw std::runtime_error("Error creating CS with lsb<0 in CompServ::CompServ");
    }

    mem_level=mem;
    if ((mem<0)||(mem>2)) {
      fprintf(stderr,"Error creating CS with incorrect memory saving level param in CompServ::CompServ");
      exit(EXIT_FAILURE);
    }
    else {
      if (mem==1) fprintf(stderr,"CS with full memory setup\n");
      if (mem==1) fprintf(stderr,"CS with mid memory setup\n");
      if (mem==2) fprintf(stderr,"CS with low memory setup\n");
    }

}

// for Distributed
CompServ::CompServ(string filename) {
     //TODO: Implement
}
