/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/descriptors/descriptors.h"
#include <math.h>


PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());
    mem_level=mem;

    this->padding = "custom";

    if (ksize.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (stride.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    //if (pad.size() != 2) msg("Padding must have 2 dimensions", "PoolDescriptor::PoolDescriptor");

}

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st, const string& p, int mem) {
    if (ks.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (st.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");

    ksize = ks;
    stride = st;
    mem_level=mem;

    if (p=="same" || p =="none" || p =="valid" || p =="zeros") {
        this->padding=p;
    }else{
        msg("Incorrect padding type", "PoolDescriptor::PoolDescriptor");
    }
}


PoolDescriptor::~PoolDescriptor(){
    delete indX;
    delete indY;
}

void PoolDescriptor::build(Tensor *A) {
    if (A->ndim != 4) msg("Tensors are not 4D", "PoolDescriptor::build");

    I = A;

    kr = ksize[0];
    kc = ksize[1];

    sr = stride[0];
    sc = stride[1];

    in = A->shape[0];
    iz = A->shape[1];
    ir = A->shape[2];
    ic = A->shape[3];

    if(this->padding=="custom"){  // Known padding
        // Compute output
        z = iz;
        r = compute_output(this->pad, ir, kr, sr);
        c = compute_output(this->pad, ic, kc, sc);

    }else{  // Common padding (same/zeros)
        // Compute output
        z = iz;
        r = compute_output(this->padding, ir, kr, sr);
        c = compute_output(this->padding, ic, kc, sc);

        // Compute padding
        vector<int> padr = compute_padding(r, ir, kr, sr, this->padding,true);  // Order: [top, bottom]
        vector<int> padc = compute_padding(c, ic, kc, sc, this->padding,false);  // Order: [left, right]

        // Set padding
        pad = {padr[0], padr[1], padc[0], padc[1]};  // top, bottom, left, right
    }

    padrt = pad[0]; padrb = pad[1];  // rows: top-bottom
    padcl = pad[2]; padcr = pad[3];  // cols: left-right

    if ((r <= 0) || (c <= 0)) {
        if(r <= 0) { std::cerr << "'Rows' are reach 0 or less (" << r << ")" << std::endl; }
        if(c <= 0) { std::cerr << "'Columns' are reach 0 or less (" << c << ")" << std::endl; }
        msg("Invalid output shape", "PoolDescriptor::build");
    }

    this->O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
//    if (!mem_level) { D = new Tensor(this->O->shape, A->device); }


    // Careful with the "size++" not "useless loop"
    size=0;
    for(int k=0;k<iz;k++)
      for(int i=-padrt;i<=ir+padrb-kr;i+=sr)
        for(int j=-padcl;j<=ic+padcr-kc;j+=sc,size++) {}

#ifdef cCUDNN
    cudnn_handle = hdnn[A->gpu_device];
    cudnnCreatePoolingDescriptor(&poolingDesc);

    windowHeight = kr;
    windowWidth = kc;
    verticalPadding = padrt;
    horizontalPadding = padcl;
    verticalStride = sr;
    horizontalStride = sc;
    // mode is initialized in each constructor.
    data_type = CUDNN_DATA_FLOAT;
    tensor_format = CUDNN_TENSOR_NCHW;  // CUDNN_TENSOR_NHWC

    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, tensor_format, data_type,
                 in,iz,ir,ic);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, tensor_format, data_type, in, z,r,c);

#endif

}

void PoolDescriptor::resize(int b) {
  if (b == this->O->shape[0]) return;

  this->O->resize(b);
#ifdef cCUDNN
    #ifdef cCUDNN
   cudnnSetTensor4dDescriptor(xDesc, tensor_format, data_type,
                 b,iz,ir,ic);

   cudnnCreateTensorDescriptor(&yDesc);
   cudnnSetTensor4dDescriptor(yDesc, tensor_format, data_type, O->shape[0], O->shape[1],O->shape[2],O->shape[3]);

#endif
#endif
//  if (!mem_level) { D->resize(b); }
}


int PoolDescriptor::compute_output(const string& padding, int input_size, int kerkel_size, int stride, int dilation_rate){
    if (padding=="same" || padding =="zeros") {
        return std::ceil((float)input_size/(float)stride);

    }else if(padding =="valid" || padding =="none"){
        return std::ceil(((float)input_size - ((float)kerkel_size - 1.0f) * (float)dilation_rate)/(float)stride);

    }else{
      cout<<padding<<endl;
        msg("Incorrect padding type", "ConvolDescriptor::compute_output");
    }
    return -1;
}

int PoolDescriptor::compute_output(vector<int> padding, int input_size, int kerkel_size, int stride, int dilation_rate) {
    return (int)(((float)input_size - ((float)kerkel_size - 1.0f) * (float)dilation_rate + (float)padding[0] + (float)padding[1] - 1.0f)/(float)stride + 1.0f);
}

vector<int> PoolDescriptor::compute_padding(int output_size, int input_size, int kerkel_size, int stride, string padding, bool row){
    // Padding order: [left, right] // [top, bottom]

    if (padding=="same,none") {
      if (row) padding="same";
      else padding="none";
    }
    if (padding=="none,same") {
      if (row) padding="none";
      else padding="same";
    }

    if (padding=="same" || padding =="zeros") {
        int pad = (output_size-1) * stride + kerkel_size - input_size;
        pad = std::max(pad, 0);

        // Ignore the padding if possible
        int padl = pad/2;  // 1/2=0.5 => 0
        int padr = pad - pad/2; // 1-1/2 = 1-0 => 1

        return vector<int>({padl, padr});

    }else if(padding =="valid" || padding =="none"){
        return vector<int>({0, 0});
    }
    else{
        cout<<padding<<endl;
        msg("Incorrect padding type", "ConvolDescriptor::compute_padding");
    }

    return {-1};
}
