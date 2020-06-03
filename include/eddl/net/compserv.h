/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_COMPSERV_H
#define EDDL_COMPSERV_H

#include <cstdio>
#include <string>
#include <vector>


using namespace std;

class CompServ {
public:
    string type;


    int local_threads;
    vector<int> local_gpus;
    vector<int> local_fpgas;
    int lsb; //local sync batches
    bool isshared;



    // memory requirements level
    // 0: full memory. better performance in terms of speed
    // 1: mid memory. some memory improvements to save memory
    // 2: low memory. save memory as much as possible
    int mem_level;



    CompServ();
    CompServ * share();

    // for local
    CompServ(int threads, const vector<int> g, const vector<int> &f,int lsb=1, int mem=0);

    // for Distributed
    explicit CompServ(string filename);


};

#endif  //EDDL_COMPSERV_H
