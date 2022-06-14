/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/descriptors/descriptors.h"
#include <cmath>


PoolDescriptor3D::PoolDescriptor3D(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());
    mem_level=mem;

    this->padding = "custom";

    if (ksize.size() != 3) msg("Pooling Kernels must have 3 dimensions", "PoolDescriptor3D::PoolDescriptor3D");
    if (stride.size() != 3) msg("Strides must have 3 dimensions", "PoolDescriptor3D::PoolDescriptor3D");
    //if (pad.size() != 3) msg("Padding must have 3 dimensions", "PoolDescriptor3D::PoolDescriptor3D");

}

PoolDescriptor3D::PoolDescriptor3D(const vector<int> &ks, const vector<int> &st, const string& p, int mem) {
    if (ks.size() != 3) msg("Pooling Kernels must have 3 dimensions", "PoolDescriptor3D::PoolDescriptor3D");
    if (st.size() != 3) msg("Strides must have 3 dimensions", "PoolDescriptor3D::PoolDescriptor3D");

    ksize = ks;
    stride = st;
    mem_level=mem;

    if (p=="same" || p =="none" || p =="valid" || p =="zeros") {
        this->padding=p;
    }else{
        msg("Incorrect padding type", "PoolDescriptor3D::PoolDescriptor3D");
    }
}


PoolDescriptor3D::~PoolDescriptor3D(){
    if( indX != nullptr) { delete indX; indX= nullptr; }
    if( indY != nullptr) { delete indY; indY= nullptr; }
    if( indZ != nullptr) { delete indZ; indZ= nullptr; }
}

void PoolDescriptor3D::build(Tensor *A) {
    if (A->ndim != 5) msg("Tensors are not 5D", "PoolDescriptor3D::build");

    I = A;

    kd = ksize[0];
    kr = ksize[1];
    kc = ksize[2];

    sd = stride[0];
    sr = stride[1];
    sc = stride[2];
    
    in = A->shape[0];
    iz = A->shape[1];
    id = A->shape[2];
    ir = A->shape[3];
    ic = A->shape[4];

    if(this->padding=="custom"){  // Known padding
        // Compute output
        z = iz;
        d = compute_output(this->pad, id, kd, sd);
        r = compute_output(this->pad, ir, kr, sr);
        c = compute_output(this->pad, ic, kc, sc);

    }else{  // Common padding (same/zeros)
        // Compute output
        z = iz;
        d = compute_output(this->padding, id, kd, sd);
        r = compute_output(this->padding, ir, kr, sr);
        c = compute_output(this->padding, ic, kc, sc);

        // Compute padding
        vector<int> padd = compute_padding(d, id, kd, sd, this->padding,true);  // Order: [front, back]
        vector<int> padr = compute_padding(r, ir, kr, sr, this->padding,true);  // Order: [top, bottom]
        vector<int> padc = compute_padding(c, ic, kc, sc, this->padding,false);  // Order: [left, right]

        // Set padding
        pad = {padd[0], padd[1], padr[0], padr[1], padc[0], padc[1]};  // (front, back), (top, bottom), (left, right)
    }

    paddf = pad[0]; paddb = pad[1];  // depth: front-back
    padrt = pad[2]; padrb = pad[3];  // rows: top-bottom
    padcl = pad[4]; padcr = pad[5];  // cols: left-right
#ifdef cCUDNN
       if(!A->isCPU()){
           if(pad[0] != pad[1] || pad[2] != pad[3] || pad[4] != pad[5]){
             msg("Warning: asymmetric padding not supported by cuDNN... fixing ... potential shapes mismatch later", "pool3d");
               //std::cout<<"Warning: asymmetric padding not supported by cuDNN... fixing ... potential shapes mismatch later"<<std::endl;
           }
           if (pad[0] != pad[1]){pad[0] = pad[1];}
           if (pad[2] != pad[3]){ pad[2] = pad[3];}
           if (pad[4] != pad[5]){ pad[4] = pad[5];}
      }
#endif

    if ((d <= 0) || (r <= 0) || (c <= 0)) {
        if(d <= 0) { std::cerr << "'Depth' are reach 0 or less (" << d << ")" << std::endl; }
        if(r <= 0) { std::cerr << "'Rows' are reach 0 or less (" << r << ")" << std::endl; }
        if(c <= 0) { std::cerr << "'Columns' are reach 0 or less (" << c << ")" << std::endl; }
        msg("Invalid output shape", "PoolDescriptor3D::build");
    }

    O = Tensor::zeros(vector<int>{A->shape[0], z, d, r, c}, A->device);
//    if (!mem_level) { D = new Tensor(O->shape, A->device); }


    // Careful with the "size++" not "useless loop"
    size=0;
    for(int k=0;k<iz;k++)
        for(int w=-paddf;w<=id+paddb-kd;w+=sd)
            for(int i=-padrt;i<=ir+padrb-kr;i+=sr)
                for(int j=-padcl;j<=ic+padcr-kc;j+=sc,size++) {}
//    cout << "Size 1: " << size << endl;

//    int w = std::floor(((id+paddb-kd)-(-paddf))/sd) + 1;  // max-min+1
//    int i = std::floor(((ir+padrb-kr)-(-padrt))/sr) + 1;  // max-min+1
//    int j = std::floor(((ic+padcr-kc)-(-padcl))/sc) + 1;  // max-min+1
//    size=iz * w *  i * j;

#ifdef cCUDNN
    if(!O->isCPU()){
    cudnnCreatePoolingDescriptor(&poolingDesc);

    cwindow[0] = kd;
    cwindow[1] = kr;
    cwindow[2] = kc;
    cpadding[0] = pad[0];
    cpadding[1] = pad[1];
    cpadding[2] = pad[2];
    cstride[0] = sd;
    cstride[1] = sr;
    cstride[2] = sc;
    // mode is initialized in each constructor.
    data_type = CUDNN_DATA_FLOAT;
    tensor_format = CUDNN_TENSOR_NCHW;  // CUDNN_TENSOR_NHWC

    cudnnCreateTensorDescriptor(&xDesc);
   int dims[5] = {in, iz, id, ir, ic};
   int str[5] = {iz*id*ir*ic,id*ir*ic,ir*ic,ic,1};
   cudnnSetTensorNdDescriptor(xDesc, data_type,5,dims,str);

   int ydims[5] = {in,z,d,r,c};
   int ystr[5] = {z*d*r*c, d*r*c, r*c, c, 1};
   cudnnCreateTensorDescriptor(&yDesc);
   cudnnSetTensorNdDescriptor(yDesc, data_type, 5, ydims, ystr);
}

#endif

}

void PoolDescriptor3D::resize(int b) {
  if (b == O->shape[0]) return;

  O->resize(b);
#ifdef cCUDNN
  if(!O->isCPU()){
      cudnnDestroyTensorDescriptor(xDesc);
      cudnnDestroyTensorDescriptor(yDesc);

      cudnnCreateTensorDescriptor(&xDesc);
      cudnnCreateTensorDescriptor(&yDesc);

       int dims[5] = {b, iz, id, ir, ic};
       int str[5] = {iz*id*ir*ic,id*ir*ic,ir*ic,ic,1};
       cudnnSetTensorNdDescriptor(xDesc, data_type,5,dims,str);

       int ydims[5] = {b,z,d,r,c};
       int ystr[5] = {z*d*r*c, d*r*c, r*c, c, 1};
       cudnnSetTensorNdDescriptor(yDesc, data_type, 5, ydims, ystr);
}


#endif

//  if (!mem_level) { D->resize(b); }
}


int PoolDescriptor3D::compute_output(const string& padding, int input_size, int kernel_size, int stride, int dilation_rate){
    if (padding=="same" || padding =="zeros") {
        return std::ceil((float)input_size/(float)stride);

    }else if(padding =="valid" || padding =="none"){
        return std::ceil(((float)input_size - ((float)kernel_size - 1.0f) * (float)dilation_rate)/(float)stride);

    }else{
      cout<<padding<<endl;
        msg("Incorrect padding type", "ConvolDescriptor::compute_output");
    }
    return -1;
}

int PoolDescriptor3D::compute_output(vector<int> padding, int input_size, int kernel_size, int stride, int dilation_rate) {
    return (int)(((float)input_size - ((float)kernel_size - 1.0f) * (float)dilation_rate + (float)padding[0] + (float)padding[1] - 1.0f)/(float)stride + 1.0f);
}

vector<int> PoolDescriptor3D::compute_padding(int output_size, int input_size, int kernel_size, int stride, string padding, bool row){
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
        int pad = (output_size-1) * stride + kernel_size - input_size;
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
