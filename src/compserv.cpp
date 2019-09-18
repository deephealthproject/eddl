
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "compserv.h"


// for local
CompServ::CompServ(int t, const vector<int> &g, const vector<int> &f,int lsb) {
    type = "local";
    local_threads = t;

    local_gpus = vector<int>(g.begin(), g.end());
    local_fpgas = vector<int>(f.begin(), f.end());

    this->lsb=lsb;
    if (lsb<0) {
      fprintf(stderr,"Error creating CS with lsb<0 in CompServ::CompServ");
      exit(EXIT_FAILURE);
    }
}

// for Distributed
CompServ::CompServ(FILE *csspec) {
     int asd = 33;
}
