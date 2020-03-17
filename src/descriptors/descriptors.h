/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_DESCRIPTORS_H
#define EDDL_DESCRIPTORS_H

#include <stdio.h>
#include <vector>
#include <string>
#include <mutex>

#include <Eigen/Dense>
#include "../tensor/tensor.h"

#include "../utils.h"


using namespace std;

class MapReduceDescriptor {
public:
   int *ind;
   int *gind;


   MapReduceDescriptor(Tensor *A,vector<int> axis);
   ~MapReduceDescriptor();
};

class ReduceDescriptor {
public:
   vector<int> axis;
   bool keepdims;
   int m;
   int red_size;

   vector<vector<int>> index;
   Tensor *I; // input
   Tensor *O; // output
   Tensor *D; // delta
   Tensor *ID; // parent delta
   Tensor *S; // indexes for max,min...
   // for gpu:
   int *ind;
   float *red;
   int factor;

   ReduceDescriptor();
   ReduceDescriptor(Tensor *A,vector<int> axis, string mode, bool keepdims);
   void resize(int b);
   void build_index();

};

class ConvolDescriptor {
public:
    vector<int> ksize;
    vector<int> stride;
    vector<int> pad; // {rows-top, rows-bottom, cols-left, cols-right}
    string padding; // valid/none, same/zeros, custom

    int nk, kr, kc, kz;
    int sr, sc;
    int ir, ic, iz;
    int r, c, z;
    int padrt,padrb;
    int padcl,padcr;
    int size;
    bool use_bias;
    int mem_level; // see CS

    Tensor *I= nullptr; // Input map
    Tensor *ID= nullptr;// Delta input map
    Tensor *K= nullptr; // filters
    Tensor *bias= nullptr; // bias
    Tensor *gK= nullptr;// gradient filters
    Tensor *gbias= nullptr;// gradient bias
  	Tensor *acc_gK= nullptr;// Accumulated gradients for kernels
  	Tensor *acc_gbias= nullptr;// Accumulated gradients for bias
    Tensor *D = nullptr; // Delta
    Tensor *O= nullptr; // Outputmap

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
    Tensor *gpuOB; // output
    Tensor *gpuK; // kernels
    Tensor *gpugK; // gradient kernels
    Tensor *gpuD; // Delta

    ConvolDescriptor();

    ConvolDescriptor(int filters, const vector<int> &ks, const vector<int> &st, const string& p, bool use_bias, int mem=0);

    ConvolDescriptor(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem=0);

    void build(Tensor *A);
    void resize(int b);
	void enable_distributed();

	static int compute_output(const string& padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
	static int compute_output(vector<int> padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static vector<int> compute_padding(int output_size, int input_size, int kerkel_size, int stride, string padding="same");

    };


class PoolDescriptor : public ConvolDescriptor {
public:
    Tensor *indX, *indY; // indexes
    int mem_level; // see CS

    PoolDescriptor(const vector<int> &ks, const vector<int> &st, const string& p, int mem=0);

    PoolDescriptor(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem=0);

    void build(Tensor *A);
    void resize(int b);
};

#endif //EDDL_DESCRIPTORS_H
