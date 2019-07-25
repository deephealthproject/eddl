
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

#ifndef EDDLL_COMPSERV_H
#define EDDLL_COMPSERV_H

#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

class CompServ {
public:
    string type;


    int local_threads;
    vector<int> local_gpus;
    vector<int> local_fpgas;

    // for local
    CompServ(int threads, const vector<int> &g, const vector<int> &f);

    // for Distributed
    explicit CompServ(FILE *csspec);

};

#endif  //EDDLL_COMPSERV_H
