/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_DESCRIPTORS_H
#define EDDL_DESCRIPTORS_H

#include <cstdio>
#include <vector>
#include <string>
#include <mutex>

#include "Eigen/Dense"

#include "eddl/tensor/tensor.h"
#include "eddl/utils.h"
#ifdef cCUDNN
#include <cudnn.h>
extern cudnnHandle_t hdnn;

#endif

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

    ~ReduceDescriptor();

    void resize(int b);
    void build_index();

};

class ConvolDescriptor {
public:
    vector<int> ksize;
    vector<int> stride;
    vector<int> pads; // {rows-top, rows-bottom, cols-left, cols-right}

    int nk, kr, kc, kz;
    int sr, sc;
    int in, ir, ic, iz;
    int r, c, z;
    int padrt,padrb;
    int padcl,padcr;
    int size;  // Auxiliar var


    // To store info...
    int filters;
    vector<int> kernel_size;
    vector<int> strides;
    string padding; // valid/none, same/zeros, custom
    int groups = groups;
    vector<int> dilation_rate;
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

#ifdef cCUDNN
    // Following cuDNN nomenclature
    cudnnHandle_t cudnn_handle;
    cudnnConvolutionMode_t convolution_mode;
    cudnnDataType_t data_type;
    cudnnTensorFormat_t tensor_format;

    cudnnConvolutionFwdAlgo_t fwd_algorithm;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algorithm;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algorithm;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnTensorDescriptor_t xDesc; //input. also dxDesc
    cudnnFilterDescriptor_t wDesc; //kernels also dwDesc
    cudnnTensorDescriptor_t yDesc; //output also dyDesc
    cudnnTensorDescriptor_t bDesc; //bias, also dbias

    int cudnn_env_init;
    int cudnn_conv_back_init;
#endif



#ifdef cFPGA
    // FPGA implementation
    cl::Buffer *fpga_ptrI;
    long int fpga_sizeI;
#endif

    ConvolDescriptor();

    ConvolDescriptor(int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding, const vector<int> &pads,
                     int groups, const vector<int> &dilation_rate, bool use_bias, int mem=0);

    ~ConvolDescriptor();

    void build(Tensor *A);
    void resize(int b);
    void enable_distributed();

    static int compute_output(const string& padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static int compute_output(vector<int> padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static vector<int> compute_padding(int output_size, int input_size, int kerkel_size, int stride, string padding="same",bool row=false);

};

class ConvolDescriptor3D {
public:
    vector<int> ksize;
    vector<int> stride;
    vector<int> pad; // {depth-front, depth-back, rows-top, rows-bottom, cols-left, cols-right}
    string padding; // valid/none, same/zeros, custom

    int nk, kz, kd, kr, kc;  // nk=num filters, kz=kernel channels, kd=kernel depth, kr=kernel rows, kc=Kernel cols
    int sd, sr, sc;  // sd=stride depth, sr=stride rows, sc=stride cols
    int in, iz, id, ir, ic;  // in=input batches, iz=input channels, id=input depth, ir=input rows, ic=input cols
    int z, d, r, c;  // z=channels, d=depth, r=rows, c=cols
    int paddf,paddb;  // pad(ding) d(epth) + f(ront) / b(ack)
    int padrt,padrb; // pad(ding) r(ows) + t(op) / b(ottom)
    int padcl,padcr; // pad(ding) c(ols) + l(eft) / r(ight)
    int size;  // Auxiliar var
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

#ifdef cFPGA
    // FPGA implementation
    cl::Buffer *fpga_ptrI;
    long int fpga_sizeI;
#endif

    ConvolDescriptor3D();

    ConvolDescriptor3D(int filters, const vector<int> &ks, const vector<int> &st, const string& p, bool use_bias, int mem=0);

    ConvolDescriptor3D(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem=0);

    ~ConvolDescriptor3D();

    void build(Tensor *A);
    void resize(int b);
    void enable_distributed();

    static int compute_output(const string& padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static int compute_output(vector<int> padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static vector<int> compute_padding(int output_size, int input_size, int kerkel_size, int stride, string padding="same",bool row=false);

};

class PoolDescriptor {

public:
    Tensor *indX, *indY; // indexes
    vector<int> ksize;
    vector<int> stride;
    vector<int> pad; // {rows-top, rows-bottom, cols-left, cols-right}
    string padding; // valid/none, same/zeros, custom

    int nk, kr, kc, kz;
    int sr, sc;
    int in, ir, ic, iz;
    int r, c, z;
    int padrt,padrb;
    int padcl,padcr;
    int size;  // Auxiliar var
    bool use_bias;
    int mem_level; // see CS

    Tensor *I= nullptr; // Input map
    Tensor *ID= nullptr;// Delta input map
    Tensor *D = nullptr; // Delta
    Tensor *O= nullptr; // Outputmap

#ifdef cCUDNN
    cudnnPoolingDescriptor_t    poolingDesc;
    cudnnPoolingMode_t          mode;
    cudnnNanPropagation_t       maxpoolingNanOpt;
    int                         windowHeight;
    int                         windowWidth;
    int                         verticalPadding;
    int                         horizontalPadding;
    int                         verticalStride;
    int                         horizontalStride;
/* Heritage from convolution descriptor
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t xDesc; //input. also dxDesc
    cudnnTensorDescriptor_t yDesc; //output also dyDesc
*/
#endif

    PoolDescriptor(const vector<int> &ks, const vector<int> &st, const string& p, int mem=0);

    PoolDescriptor(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem=0);

    ~PoolDescriptor();

    void build(Tensor *A);
    void resize(int b);
    static int compute_output(const string& padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static int compute_output(vector<int> padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static vector<int> compute_padding(int output_size, int input_size, int kerkel_size, int stride, string padding="same",bool row=false);
};

class PoolDescriptor3D {

public:
    Tensor *indX, *indY; // indexes
    vector<int> ksize;
    vector<int> stride;
    vector<int> pad; // {depth-front, depth-back, rows-top, rows-bottom, cols-left, cols-right}
    string padding; // valid/none, same/zeros, custom

    int nk, kz, kd, kr, kc;  // nk=num filters, kz=kernel channels, kd=kernel depth, kr=kernel rows, kc=Kernel cols
    int sd, sr, sc;  // sd=stride depth, sr=stride rows, sc=stride cols
    int iz, id, ir, ic;  // iz=input channels, id=input depth, ir=input rows, ic=input cols
    int z, d, r, c;  // z=channels, d=depth, r=rows, c=cols
    int paddf,paddb;  // pad(ding) d(epth) + f(ront) / b(ack)
    int padrt,padrb; // pad(ding) r(ows) + t(op) / b(ottom)
    int padcl,padcr; // pad(ding) c(ols) + l(eft) / r(ight)
    int size;  // Auxiliar var
    bool use_bias;
    int mem_level; // see CS

    Tensor *I= nullptr; // Input map
    Tensor *ID= nullptr;// Delta input map
    Tensor *D = nullptr; // Delta
    Tensor *O= nullptr; // Outputmap

    PoolDescriptor3D(const vector<int> &ks, const vector<int> &st, const string& p, int mem=0);

    PoolDescriptor3D(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem=0);

    ~PoolDescriptor3D();

    void build(Tensor *A);
    void resize(int b);
    static int compute_output(const string& padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static int compute_output(vector<int> padding, int input_size, int kerkel_size, int stride, int dilation_rate=1);
    static vector<int> compute_padding(int output_size, int input_size, int kerkel_size, int stride, string padding="same",bool row=false);
};

#endif //EDDL_DESCRIPTORS_H
