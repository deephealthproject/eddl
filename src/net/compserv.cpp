/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>

#include <stdexcept>
#include <iostream>
#include "eddl/net/compserv.h"

CompServ::CompServ()= default;


// for local
CompServ::CompServ(int threads, const vector<int>& gpus, const vector<int> &fpgas, int lsb, int mem) {
    // Set parameters
    this->type = "local";
    this->isshared = false;
    this->threads_arg = threads;
    this->lsb = lsb;
    this->mem_level = mem;

    // Get supported hardware
    this->hw_supported = {"cpu"};
    #ifdef cGPU
        this->hw_supported.emplace_back("cuda");
    #ifdef cCUDNN
        this->hw_supported.emplace_back("cudnn");
    #endif
    #endif
    #ifdef cFPGA
        this->hw_supported.emplace_back("fpga");
    #endif

    // Check: Threads
    if (this->threads_arg == -1){ this->local_threads = (int)std::thread::hardware_concurrency(); }
    else { this->local_threads = threads; }

    // Add devices
    for (auto _ : gpus) this->local_gpus.push_back(_);
    for (auto _ : fpgas) this->local_fpgas.push_back(_);

    // Check: target device
    if (!this->local_fpgas.empty()){ this->hw="fpga"; }
    else if (!this->local_gpus.empty()){ this->hw="gpu"; }
    else{ this->hw="cpu"; }

    // Check: Synchronization value
    if (this->lsb < 0) {
        throw std::runtime_error("Error creating CS with lsb<0 in CompServ::CompServ");
    }

    // Check: memory level
    if ((this->mem_level < 0) || (this->mem_level > 2)) {
        std::cerr << "Error creating CS with incorrect memory saving level param in CompServ::CompServ" << std::endl;
        exit(EXIT_FAILURE);
    }else {
        if (this->mem_level==0) { std::cerr << "CS with full memory setup" << std::endl; }
        else if (this->mem_level==1) { std::cerr << "CS with mid memory setup" << std::endl; }
        else if (this->mem_level==2) { std::cerr << "CS with low memory setup" << std::endl; }
    }

    // Check: Max device supported
    string hw_value = this->hw;
    if(hw_value=="gpu") { hw_value = "cuda"; }  // gpu could be both "cuda" and "cudnn"
    bool hw_found = std::find(this->hw_supported.begin(), this->hw_supported.end(), hw_value) != this->hw_supported.end();
    if (!hw_found){
        throw std::runtime_error("[Hardware not supported]: This library is not compiled for '" + this->hw + "'");
    }

}

CompServ* CompServ::share() {
    auto *n = new CompServ(threads_arg,local_gpus,local_fpgas,lsb,mem_level);
    n->isshared = true;
    return n;
}
CompServ* CompServ::clone() {
    auto *n = new CompServ(threads_arg,local_gpus,local_fpgas,lsb,mem_level);
    return n;
}

// for Distributed
CompServ::CompServ(const string& filename) {
    std::cerr << "Not implemented error [Computing service with filename]" << std::endl;
}

