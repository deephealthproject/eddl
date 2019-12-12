/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_TENSOR_DESCRIPTORS_H
#define EDDL_TENSOR_DESCRIPTORS_H

#include <cstdio>
#include <vector>
#include <string>
#include <mutex>

#include <Eigen/Dense>


using namespace std;


class TensorDescriptor {
public:
    TensorDescriptor(){};

    // Don't mark as pure virtual because not all methods use the same parameters
    virtual void build(){};
    virtual void resize(int b){};
};

class SelDescriptor : public TensorDescriptor{

public:
    vector<int> ishape;
    vector<int> oshape;
    vector<vector<int>> idxs_range;
    int* addresses = nullptr;
    int* gpu_addresses = nullptr;

    vector<string> indices;

    SelDescriptor();
    SelDescriptor(const vector<string>& indices);
    ~SelDescriptor();

    virtual void build(vector<int> ishape);
    virtual void resize(int b);
    virtual void build_indices();
};

class PermuteDescriptor : public SelDescriptor {
public:
    vector<int> dims;

    PermuteDescriptor(const vector<int>& dims);
    
    void build(vector<int> ishape);
    void resize(int b);
    void build_indices();
};


#endif //EDDL_TENSOR_DESCRIPTORS_H
