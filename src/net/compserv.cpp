/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <thread>

#include <stdexcept>
#include "eddl/net/compserv.h"

CompServ::CompServ()
{

}

// for local
CompServ::CompServ(int t, const vector<int> g, const vector<int> &f, int lsb, int mem) {
    type = "local";
    isshared = false;

    threads_arg = t;
    if (t == -1) local_threads = std::thread::hardware_concurrency();  // Avoid eigen dependency
    else local_threads = t;

    for (auto _ : g) this->local_gpus.push_back(_);
    for (auto _ : f) this->local_fpgas.push_back(_);

    this->lsb = lsb;

    if (lsb < 0) {
      throw std::runtime_error("Error creating CS with lsb<0 in CompServ::CompServ");
    }

    mem_level = mem;
    if ((mem < 0) || (mem > 2)) {
      fprintf(stderr,"Error creating CS with incorrect memory saving level param in CompServ::CompServ");
      exit(EXIT_FAILURE);
    }
    else {
      if (mem==0) fprintf(stderr,"CS with full memory setup\n");
      if (mem==1) fprintf(stderr,"CS with mid memory setup\n");
      if (mem==2) fprintf(stderr,"CS with low memory setup\n");
    }
}

CompServ * CompServ::share() {
  CompServ *n = new CompServ();

  n->type = this->type;
  n->local_threads = this->local_threads;
  for (auto _ : this->local_gpus) n->local_gpus.push_back(_);
  for (auto _ : this->local_fpgas) n->local_fpgas.push_back(_);
  n->lsb = this->lsb;
  n->isshared = true;
  n->mem_level = this->mem_level;

  return n;
}


// for Distributed
CompServ::CompServ(string filename) {
     //TODO: Implement
}
