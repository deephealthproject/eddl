/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/descriptors/descriptors.h"
#include <cmath>
#include <algorithm>

#include "eddl/hardware/cpu/cpu_profile.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#endif

ConvolDescriptor3D::ConvolDescriptor3D() {}

ConvolDescriptor3D::ConvolDescriptor3D(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());
    mem_level=mem;

    this->padding = "custom";

    if (ksize.size() != 4) msg("Kernels must have 4 dimensions", "ConvolDescriptor3D::ConvolDescriptor3D");
    if (stride.size() != 3) msg("Strides must have 3 dimensions", "ConvolDescriptor3D::ConvolDescriptor3D");
}

ConvolDescriptor3D::ConvolDescriptor3D(int filters, const vector<int> &ks, const vector<int> &st, const string& p, bool ub, int mem) {
    if (ks.size() != 3) { msg("Kernels must have 4 dimensions", "ConvolDescriptor3D::ConvolDescriptor3D"); }
    if (st.size() != 3) { msg("Strides must have 3 dimensions", "ConvolDescriptor3D::ConvolDescriptor3D"); }

    // Add filters to kernel_size
    ksize = vector<int>(ks);
    ksize.insert(ksize.begin(), 1, filters);
    stride = vector<int>(st.begin(), st.end());
    use_bias=ub;
    mem_level=mem;

    if (p=="same" || p =="none" || p =="valid" || p =="zeros" || p=="same,none" || p=="none,same") {
        this->padding=p;
    }else{
        cout<<p<<endl;
        msg("Incorrect padding type", "ConvolDescriptor3D::ConvolDescriptor3D");
    }

}

ConvolDescriptor3D::~ConvolDescriptor3D(){
    // input, output, delta, params[], and gradients[], acc_gradients[] => deleted in ~Layer()
    if (O->isCPU()) {
        eddl_free(ptrI);
    }
#ifdef cGPU
#ifndef cCUDNN

    else if (O->isGPU()) {

        if (mem_level>1) {
            // Lowering
            delete gpuIB;
        }
        else {
            // Big tensor with all the batch for lowering
            delete gpuIB;
            if (mem_level==0)
                delete gpuOB;
        }
    }
#endif
#endif

}

void ConvolDescriptor3D::build(Tensor *A) {

    if (A->ndim != 5) msg("Tensors are not 5D", "ConvolDescriptor3D::build");

    I = A;

    nk = ksize[0];
    kz = A->shape[1];
    kd = ksize[1];
    kr = ksize[2];
    kc = ksize[3];

    sd = stride[0];
    sr = stride[1];
    sc = stride[2];

    iz = A->shape[1];
    id = A->shape[2];
    ir = A->shape[3];
    ic = A->shape[4];

    if(this->padding=="custom"){  // Known padding
        // Compute output
        z = nk;

        vector<int>pd; pd.push_back(pad[0]);pd.push_back(pad[1]);
        d = compute_output(pd, id, kd, sd);

        vector<int>pr; pr.push_back(pad[2]);pr.push_back(pad[3]);
        r = compute_output(pr, ir, kr, sr);

        vector<int>pc; pc.push_back(pad[4]);pc.push_back(pad[5]);
        c = compute_output(pc, ic, kc, sc);

    }else{  // Common padding (same/zeros)
        // Compute output

        // Channels
        z = nk;

        // Depth
        if (padding=="same,none") d = compute_output("same", id, kd, sd);
        else if (padding=="none,same")  d = compute_output("none", id, kd, sd);
        else d = compute_output(this->padding, id, kd, sd);

        // Rows
        if (padding=="same,none") r = compute_output("same", ir, kr, sr);
        else if (padding=="none,same")  r = compute_output("none", ir, kr, sr);
        else r = compute_output(this->padding, ir, kr, sr);

        // Cols
        if (padding=="same,none") c = compute_output("none", ic, kc, sc);
        else if (padding=="none,same")  c = compute_output("same", ic, kc, sc);
        else c = compute_output(this->padding, ic, kc, sc);

        // Compute padding
        vector<int> padd = compute_padding(d, id, kd, sd, this->padding,true);  // Order: [front, back]
        vector<int> padr = compute_padding(r, ir, kr, sr, this->padding,true);  // Order: [top, bottom]
        vector<int> padc = compute_padding(c, ic, kc, sc, this->padding,false);  // Order: [left, right]

        // Set padding
        pad = {padd[0], padd[1], padr[0], padr[1], padc[0], padc[1]};  // (front, back), (top, bottom), (left, right)
    }

#ifdef cCUDNN
       if(!A->isCPU()){
           if(pad[0] != pad[1] || pad[2] != pad[3] || pad[4] != pad[5]){
             std::cout<<"Warning: asymmetric padding not supported by cuDNN... fixing ... potential shapes mismatch later"<<std::endl;
           }
           if (pad[0] != pad[1]){ pad[0] = pad[1];}
           if (pad[2] != pad[3]){ pad[2] = pad[3];}
           if (pad[4] != pad[5]){ pad[4] = pad[5];}

      }
#endif


    paddf = pad[0]; paddb = pad[1];  // depth: front-top
    padrt = pad[1]; padrb = pad[2];  // rows: top-bottom
    padcl = pad[3]; padcr = pad[4];  // cols: left-right

    if ((d <= 0) || (r <= 0) || (c <= 0)) {
        if(d <= 0) { std::cerr << "'Depth' are reach 0 or less (" << d << ")" << std::endl; }
        if(r <= 0) { std::cerr << "'Rows' are reach 0 or less (" << r << ")" << std::endl; }
        if(c <= 0) { std::cerr << "'Columns' are reach 0 or less (" << c << ")" << std::endl; }
        msg("Invalid output shape", "ConvolDescriptor3D::build");
    }

    O = new Tensor(vector<int>{A->shape[0], z, d, r, c}, A->device);

    // Params
    K = new Tensor(vector<int>{nk, kz, kd, kr, kc}, I->device);
    bias = new Tensor(vector<int>{nk}, I->device);

    gK = new Tensor(vector<int>{nk, kz, kd, kr, kc}, I->device);
    gbias = new Tensor(vector<int>{nk}, I->device);

    if (I->isCPU()) {
        // mem for ptr, lowering im2col
        unsigned long int l_size =  (unsigned long)(A->shape[0] * d * r * c) * (unsigned long)(kz * kd* kr * kc);
        ptrI=get_fmem(l_size,"ConvolDescriptor3D::build");
        matI=Eigen::Map<Eigen::MatrixXf>(ptrI, d*r*c,kz*kd*kr*kc);
	   _profile_add_tensor(A->shape[0] * d * r * c * kz * kd * kr * kc);
    }
#ifdef cGPU
    else if (I->isGPU()) {
#ifndef cCUDNN
        if (mem_level>1) {
            // Lowering
            gpuIB=new Tensor(vector<int>{d*r*c,kz*kd*kc*kr}, I->device);
        }
        else {
            // Big tensor with all the batch for lowering
            gpuIB=new Tensor(vector<int>{A->shape[0]*r*c,kz*kd*kc*kr}, I->device);
            if (mem_level==0)
                gpuOB=new Tensor(vector<int>{z,A->shape[0]*d*r*c}, I->device);
        }
#endif
        // Tensor with variable shared ptr, delete create ptr
        gpuI=new Tensor(vector<int>{d*r*c,kd*kz*kc*kr}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpuI->ptr);

        gpuO=new Tensor(vector<int>{z,d*r*c}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpuO->ptr);

        //gpu_delete_tensor(gpuI->gpu_device,gpuOB->ptr);
        gpuD=new Tensor(vector<int>{z,d*r*c}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpuD->ptr);


        gpuK=new Tensor(vector<int>{z, kz*kd*kc*kr}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpuK->ptr);
        gpugK=new Tensor(vector<int>{z, kz*kd*kc*kr}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpugK->ptr);
    }
#ifdef cCUDNN
    //CUDNN
    convolution_mode = CUDNN_CONVOLUTION; //CUDNN_CROSS_CORRELATION
    data_type = CUDNN_DATA_FLOAT;
    tensor_format = CUDNN_TENSOR_NCHW;  // CUDNN_TENSOR_NHWC

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    int padding[3] = {pad[0],pad[2],pad[4]};
    int strides[3] ={sd,sr,sc};
    int dilats[3] = {1,1,1};
    cudnnSetConvolutionNdDescriptor(convolution_descriptor,3,
                                    padding,
                                    strides,
                                    dilats,
                                    convolution_mode, data_type);


   cudnnCreateTensorDescriptor(&xDesc);
   int dims[5] = {in, iz, id, ir, ic};
   int str[5] = {iz*id*ir*ic,id*ir*ic,ir*ic,ic,1};
   cudnnSetTensorNdDescriptor(xDesc, /*tensor_format,*/ data_type,5,dims,str);

   int ydims[5] = {in,z,d,r,c};
   int ystr[5] = {z*d*r*c, d*r*c, r*c, c, 1};
   cudnnCreateTensorDescriptor(&yDesc);
   cudnnSetTensorNdDescriptor(yDesc,/* tensor_format,*/ data_type, 5, ydims, ystr);
   
   int bdims[5] = {1,z,1,1,1};
   int bstr[5] = {z, 1, 1, 1, 1};
   cudnnCreateTensorDescriptor(&bDesc);
   cudnnSetTensorNdDescriptor(bDesc,/* tensor_format,*/ data_type, 5, bdims, bstr);
   
   int fdims[5] = {nk, kz, kd, kr, kc};
  // int fstr[5] = {kz*kd*kr*kc,kd*kr*kc,kr*kc,kc,1};
   cudnnCreateFilterDescriptor(&wDesc);
   cudnnSetFilterNdDescriptor(wDesc, data_type, tensor_format, 5, fdims);

   cudnn_env_init = -1;
   cudnn_conv_back_init = -1;

#endif

#endif

#ifdef cFPGA
    if (I->isFPGA()) {
	// We allocate memory on the FGPA for the im2col buffer
	fpga_sizeI = A->shape[0] * r * c * kr * kc * kz * sizeof(float);
	fpga_ptrI = fpga_create_memory(fpga_sizeI);
	// We allocate also on cpu so to ease the cpuemu flow
        // mem for ptr, lowering im2col
        ptrI=get_fmem(A->shape[0] * d * r * c * kz * kd * kr * kc,"ConvolDescriptor3D::build");
    }
#endif
}

void ConvolDescriptor3D::resize(int b)
{
    if (b==O->shape[0]) return;

    O->resize(b);


    // Prevent overflow. (512*512*512*512*3*3*3*3 = 3,623,878,656 > MAX_INT (2,147,483,647))
    unsigned long int l_size =  (unsigned long)(b * d * r * c) * (unsigned long)(kz * kd * kr * kc);

    if (I->isCPU()) {
        eddl_free(ptrI);
        ptrI=get_fmem(l_size, "ConvolDescriptor3D::build");
	   _profile_add_tensor(l_size);
    }
#ifdef cGPU
    else if (I->isGPU()) {
#ifndef cCUDNN
        if (mem_level<2)
            gpuIB->resize(b*d*r*c);
        if (mem_level==0) {
            delete gpuOB;
            gpuOB=new Tensor(vector<int>{z,b*d*r*c}, I->device);
        }
#endif

#ifdef cCUDNN
   int dims[5] = {b, iz, id, ir, ic};
   int str[5] = {iz*id*ir*ic,id*ir*ic,ir*ic,ic,1};
   cudnnSetTensorNdDescriptor(xDesc, /*tensor_format,*/ data_type,5,dims,str);

   int ydims[5] = {b,z,d,r,c};
   int ystr[5] = {z*d*r*c, d*r*c, r*c, c, 1};
   cudnnSetTensorNdDescriptor(yDesc, /*tensor_format,*/ data_type, 5, ydims, ystr);

   //cudnnSetTensor4dDescriptor(yDesc, tensor_format, data_type, O->shape[0], O->shape[1],O->shape[2],O->shape[3]);



#endif
    }
#endif

#ifdef cFPGA
    else if (I->isFPGA()) {
        // We reallocate memory on the FGPA for the im2col buffer
	fpga_destroy_memory(fpga_ptrI);
	fpga_sizeI = l_size * sizeof(float);
        fpga_ptrI = fpga_create_memory(fpga_sizeI);
        // We do the same on the CPU side (for smooth cpuemu)
	    eddl_free(ptrI);
        ptrI=get_fmem(l_size, "ConvolDescriptor3D::build");
    }
#endif


}

void ConvolDescriptor3D::enable_distributed() {
    // Create and initialize the tensors for accumulating gradients in distributed training
    acc_gK = new Tensor(vector<int>{nk, kz, kd, kr, kc}, I->device);
    acc_gK->fill_(0.0);
    acc_gbias = new Tensor(vector<int>{nk}, I->device);
    acc_gbias->fill_(0.0);
}

int ConvolDescriptor3D::compute_output(const string& padding, int input_size, int kerkel_size, int stride, int dilation_rate){
    if (padding=="same" || padding =="zeros") {
        return std::ceil((float)input_size/(float)stride);

    }else if(padding =="valid" || padding =="none"){
        return std::ceil(((float)input_size - ((float)kerkel_size - 1.0f) * (float)dilation_rate)/(float)stride);

    }else{
      cout<<padding<<endl;
        msg("Incorrect padding type", "ConvolDescriptor3D::compute_output");
    }
    return -1;
}

int ConvolDescriptor3D::compute_output(vector<int> padding, int input_size, int kerkel_size, int stride, int dilation_rate) {
    return (int)(((float)input_size - ((float)kerkel_size - 1.0f) * (float)dilation_rate + (float)padding[0] + (float)padding[1] - 1.0f)/(float)stride + 1.0f);
}

vector<int> ConvolDescriptor3D::compute_padding(int output_size, int input_size, int kerkel_size, int stride, string padding, bool row){
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
        return vector<int>({0, 0, 0});
    }
    else{
        cout<<padding<<endl;
        msg("Incorrect padding type", "ConvolDescriptor3D::compute_padding");
    }

    return {-1};
}
