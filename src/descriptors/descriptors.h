//
// Created by Salva Carrión on 26/09/2019.
//

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
