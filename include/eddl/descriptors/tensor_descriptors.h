/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_TENSOR_DESCRIPTORS_H
#define EDDL_TENSOR_DESCRIPTORS_H

#include <cstdio>
#include <vector>
#include <string>
#include <mutex>


using namespace std;


class TensorDescriptor {
public:
    int device;

    int* cpu_addresses;
    int* gpu_addresses;
    int* fpga_addresses;

    TensorDescriptor(int dev=0);
    ~TensorDescriptor();

    // Don't mark as pure virtual because not all methods use the same parameters
    //virtual void build(){};
    virtual void resize(int b){};
    void free_memory();
};

class SelDescriptor : public TensorDescriptor {

public:
    vector<int> ishape;
    vector<int> oshape;
    vector<vector<int>> idxs_range;


    vector<string> indices;

    SelDescriptor(int dev);
    SelDescriptor(const vector<string>& indices, int dev=0);

    virtual void build(vector<int> ishape);
    void resize(int b) override;
    virtual void build_indices();
};

class PermuteDescriptor : public SelDescriptor {
public:
    vector<int> dims;

    PermuteDescriptor(const vector<int>& dims, int dev=0);

    void build(vector<int> ishape) override;
    void resize(int b) override;
    void build_indices() override;
};


#endif //EDDL_TENSOR_DESCRIPTORS_H
