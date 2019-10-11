/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_CONV_DESCRIPTOR_H
#define EDDL_CONV_DESCRIPTOR_H

#include <stdio.h>
#include <vector>
#include <string>
#include <mutex>

#include <Eigen/Dense>
#include "../tensor/tensor.h"
#include "../utils.h"


using namespace std;

class ReduceDescriptor {
public:
   vector<int> axis;
   bool keepdims;
   int m;
   int red_size;

   vector<vector<int>> index;
   int *ind;
   Tensor *I; // input
   Tensor *O; // output
   Tensor *D; // delta
   Tensor *ID; // parent delta
   Tensor *S; // indexes for max,min...


   ReduceDescriptor();
   ReduceDescriptor(Tensor *A,vector<int> axis, string mode, bool keepdims);
   void resize(int b);
   void build_index();

};

class ConvolDescriptor {
public:
    vector<int> ksize;
    vector<int> stride;
    vector<int> pad;

    int nk, kr, kc, kz;
    int sr, sc;
    int ir, ic, iz;
    int r, c, z;
    int padr, padc;
    //float *ptr;


    Tensor *I; // Input map
    Tensor *ID;// Delta input map
    Tensor *K; // filters
    Tensor *bias; // bias
    Tensor *gK;// gradient filters
    Tensor *gbias;// gradient bias
    Tensor *D; // Delta
    Tensor *O; // Outputmap

    // CPU implementation
    float *ptrI;
    Eigen::MatrixXf matI; // input
    Eigen::MatrixXf matK; // kernels
    Eigen::MatrixXf matO; // output
    Eigen::MatrixXf matD; // Delta
    Eigen::MatrixXf matgK; // gradient kernels

    // GPU implementation
    Tensor *gpuI; // input
    Tensor *gpuIB; // input
    Tensor *gpuO; // output
    Tensor *gpuK; // kernels
    Tensor *gpugK; // gradient kernels
    Tensor *gpuD; // Delta

    ConvolDescriptor();

    ConvolDescriptor(int filters, const vector<int> &ks, const vector<int> &st, string p);

    ConvolDescriptor(const vector<int> &ks, const vector<int> &st, const vector<int> &p);

    void build(Tensor *A);
    void resize(int b);
};


class PoolDescriptor : public ConvolDescriptor {
public:
    Tensor *indX, *indY; // indexes

    PoolDescriptor(const vector<int> &ks, const vector<int> &st, string p);

    PoolDescriptor(const vector<int> &ks, const vector<int> &st, const vector<int> &p);

    void build(Tensor *A);
    void resize(int b);
};

#endif //EDDL_CONV_DESCRIPTOR_H
