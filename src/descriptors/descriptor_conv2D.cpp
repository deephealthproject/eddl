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

ConvolDescriptor::ConvolDescriptor() {}

ConvolDescriptor::ConvolDescriptor(int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding, const vector<int> &pads,
                 int groups, const vector<int> &dilation_rate, bool use_bias, int mem){
#ifndef cCUDNN
    if (groups > 1) { msg("Grouped convolutions are only available with CuDNN", "ConvolDescriptor::ConvolDescriptor"); }
#endif
    if (kernel_size.size() != 2) { msg("Kernels must have 3 dimensions", "ConvolDescriptor::ConvolDescriptor"); }
    if (strides.size() != 2) { msg("Strides must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor"); }
    if (dilation_rate.size() != 2) { msg("Dilations must have 2 elements", "ConvolDescriptor::ConvolDescriptor"); }
    if (groups < 1) { msg("The number of groups should be greater or equal to 1", "ConvolDescriptor::ConvolDescriptor"); }

    // Store stuff
    this->filters = filters;
    this->kernel_size = kernel_size;
    this->strides = strides;
    this->padding = padding;  // none, same
    this->pads = vector<int>(pads);  // (1,1,1,1)
    this->groups = groups;
    this->dilation_rate = dilation_rate;
    this->use_bias=use_bias;
    this->mem_level=mem;

    // Add filters to kernel_size (old)
    ksize = vector<int>(kernel_size);
    ksize.insert(ksize.begin(), 1, filters);
    stride = vector<int>(strides);

    if (!(padding == "custom" || padding=="same" || padding =="none" || padding =="valid" || padding =="zeros" || padding=="same,none" || padding=="none,same")) {
        msg("Incorrect padding type (" + padding + ")", "ConvolDescriptor::ConvolDescriptor");
    }

    // Check that the number of groups is valid
    if(filters % groups)
        msg("The number of filters must be divisible by the number groups."
            " Received: filters=" + to_string(filters) + " groups=" + to_string(groups),
            "ConvolDescriptor::ConvolDescriptor");
}


ConvolDescriptor::~ConvolDescriptor(){
    // input, output, delta, params[], and gradients[], acc_gradients[] => deleted in ~Layer()
    if (O->isCPU()) {
        eddl_free(ptrI); // because get_fmem() now uses posix_memalign()
    }
#ifdef cGPU
#ifndef cCUDNN
    else if (O->isGPU()) {

        if (mem_level == 1) {
            // Lowering
            delete gpuIB;
        } else if (mem_level == 0) {
            // Big tensor with all the batch for lowering
            delete gpuIB;
            delete gpuOB;
        }
    }
#endif
#endif

}

void ConvolDescriptor::build(Tensor *A) {

    if (A->ndim != 4) msg("Tensors are not 4D", "ConvolDescriptor::build");

    I = A;

    nk = ksize[0];
    kr = ksize[1];
    kc = ksize[2];
    if(A->shape[1] % groups)
        msg("The number of input channels must be divisible by the number groups."
            " Received: in_channels=" + to_string(A->shape[1]) + " groups=" + to_string(groups),
            "ConvolDescriptor::build");
    kz = A->shape[1] / groups;

    sr = stride[0];
    sc = stride[1];

    in = A->shape[0]; //batch size
    iz = A->shape[1];
    ir = A->shape[2];
    ic = A->shape[3];

    if(this->padding=="custom"){  // Known padding
        // Compute output
        z = nk;
        vector<int>pr; pr.push_back(pads[0]);pr.push_back(pads[1]);
        r = compute_output(pr, ir, kr, sr, dilation_rate[0]);

        vector<int>pc; pc.push_back(pads[2]);pc.push_back(pads[3]);
        c = compute_output(pc, ic, kc, sc, dilation_rate[1]);

    }else{  // Common padding (same/zeros)
        // Compute output
        z = nk;

        if (padding=="same,none") r = compute_output("same", ir, kr, sr, dilation_rate[0]);
        else if (padding=="none,same")  r = compute_output("none", ir, kr, sr, dilation_rate[0]);
        else r = compute_output(this->padding, ir, kr, sr, dilation_rate[0]);

        if (padding=="same,none") c = compute_output("none", ic, kc, sc, dilation_rate[1]);
        else if (padding=="none,same")  c = compute_output("same", ic, kc, sc, dilation_rate[1]);
        else c = compute_output(this->padding, ic, kc, sc, dilation_rate[1]);

        // Compute padding
        vector<int> padr = compute_padding(r, ir, kr, sr, this->padding,true, dilation_rate[0]);  // Order: [top, bottom]
        vector<int> padc = compute_padding(c, ic, kc, sc, this->padding,false, dilation_rate[1]);  // Order: [left, right]

        // Set padding
        this->pads.clear();
        this->pads = {padr[0], padr[1], padc[0], padc[1]};  // top, bottom, left, right
    }

#ifdef cCUDNN
       if(!A->isCPU()){
           if(pads[0] != pads[1] || pads[2] != pads[3]){
               msg("Warning: asymmetric padding not supported by cuDNN... fixing ... potential shapes mismatch later", "conv2d");
                //std::cout<<"Warning: asymmetric padding not supported by cuDNN... fixing ... potential shapes mismatch later"<<std::endl;
           }
           if (pads[0] != pads[1]){pads[0] = pads[1];}
           if (pads[2] != pads[3]){ pads[2] = pads[3];}
      }
#endif

    padrt = pads[0]; padrb = pads[1];  // rows: top-bottom
    padcl = pads[2]; padcr = pads[3];  // cols: left-right

    if ((r <= 0) || (c <= 0)) {
        if(r <= 0) { std::cerr << "'Rows' are reach 0 or less (" << r << ")" << std::endl; }
        if(c <= 0) { std::cerr << "'Columns' are reach 0 or less (" << c << ")" << std::endl; }
        msg("Invalid output shape", "ConvolDescriptor::build");
    }

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);

    // Params
    K = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    bias = new Tensor(vector<int>{nk}, I->device);

    gK = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    gbias = new Tensor(vector<int>{nk}, I->device);

    if (I->isCPU()) {
        if (mem_level < 2) {
            // mem for ptr, lowering im2col
            unsigned long int l_size =  (unsigned long)(A->shape[0] * r * c) * (unsigned long)(kr * kc * kz);
            ptrI=get_fmem(l_size,"ConvolDescriptor::build");
            matI=Eigen::Map<Eigen::MatrixXf>(ptrI, r*c,kz*kr*kc);
               _profile_add_tensor(A->shape[0] * r * c * kr * kc * kz);
        }
    }
#ifdef cGPU
    else if (I->isGPU()) {
#ifndef cCUDNN
        if (mem_level == 1) {
            // Lowering
            gpuIB=new Tensor(vector<int>{r*c,kc*kr*kz}, I->device);
        } else if (mem_level== 0) {
            // Big tensor with all the batch for lowering
            gpuIB=new Tensor(vector<int>{A->shape[0]*r*c,kc*kr*kz}, I->device);
            gpuOB=new Tensor(vector<int>{z,A->shape[0]*r*c}, I->device);
        }
#endif
        // Tensor with variable shared ptr, delete create ptr
        gpuI=new Tensor(vector<int>{r*c,kc*kr*kz}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpuI->ptr);

        gpuO=new Tensor(vector<int>{z,r*c}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpuO->ptr);

        //gpu_delete_tensor(gpuI->gpu_device,gpuOB->ptr);
        gpuD=new Tensor(vector<int>{z,r*c}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpuD->ptr);


        gpuK=new Tensor(vector<int>{z,kc*kr*kz}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpuK->ptr);
        gpugK=new Tensor(vector<int>{z,kc*kr*kz}, I->device);
        gpu_delete_tensor(gpuI->gpu_device,gpugK->ptr);
    }
#ifdef cCUDNN
    //CUDNN
    convolution_mode = CUDNN_CROSS_CORRELATION; //CUDNN_CONVOLUTION; 
    data_type = CUDNN_DATA_FLOAT;
    tensor_format = CUDNN_TENSOR_NCHW;  // CUDNN_TENSOR_NHWC

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolutionGroupCount(convolution_descriptor, groups);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    pads[0], pads[2],
                                    stride[0], stride[1],
                                    dilation_rate[0], dilation_rate[1],
                                    convolution_mode, data_type);

   cudnnCreateTensorDescriptor(&xDesc);
   cudnnSetTensor4dDescriptor(xDesc, tensor_format, data_type,
                 in,iz,ir,ic);

   cudnnCreateFilterDescriptor(&wDesc);
   cudnnSetFilter4dDescriptor(wDesc, data_type, tensor_format, nk, kz, kr, kc);

   cudnnCreateTensorDescriptor(&yDesc);
   cudnnSetTensor4dDescriptor(yDesc, tensor_format, data_type, in, z,r,c);

   cudnnCreateTensorDescriptor(&bDesc);
   cudnnSetTensor4dDescriptor(bDesc, tensor_format, data_type, 1, nk,1,1);
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
        ptrI=get_fmem(A->shape[0] * r * c * kr * kc * kz,"ConvolDescriptor::build");
    }
#endif
}

void ConvolDescriptor::resize(int b)
{
    if (b==O->shape[0]) return;

    O->resize(b);

    // Prevent overflow. (512*512*512*3*3*3 = 3,623,878,656 > MAX_INT (2,147,483,647))
    unsigned long int l_size =  (unsigned long)(b * r * c) * (unsigned long)(kr * kc * kz);

    if (I->isCPU()) {
        eddl_free(ptrI); // because get_fmem() now uses posix_memalign()
        ptrI=get_fmem(l_size, "ConvolDescriptor::build");
	   _profile_add_tensor(l_size);
    }
#ifdef cGPU
    else if (I->isGPU()) {
#ifndef cCUDNN
      if (mem_level < 1)
            gpuIB->resize(b*r*c);
        if (mem_level==0) {
            delete gpuOB;
            gpuOB=new Tensor(vector<int>{z,b*r*c}, I->device);
        }
#endif

#ifdef cCUDNN
   cudnnSetTensor4dDescriptor(xDesc, tensor_format, data_type,
                 b,iz,ir,ic);

   cudnnCreateTensorDescriptor(&yDesc);
   cudnnSetTensor4dDescriptor(yDesc, tensor_format, data_type, O->shape[0], O->shape[1],O->shape[2],O->shape[3]);

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
        eddl_free(ptrI); // because get_fmem() now uses posix_memalign()
        ptrI=get_fmem(l_size, "ConvolDescriptor::build");
    }
#endif


}

void ConvolDescriptor::enable_distributed() {
    // Create and initialize the tensors for accumulating gradients in distributed training
    acc_gK = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    acc_gK->fill_(0.0);
    acc_gbias = new Tensor(vector<int>{nk}, I->device);
    acc_gbias->fill_(0.0);
}

int ConvolDescriptor::compute_output(const string& padding, int input_size, int kernel_size, int stride, int dilation_rate){
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

int ConvolDescriptor::compute_input(const string& padding, int output_size, int kernel_size, int stride, int dilation_rate){
    if (padding=="same" || padding =="zeros") {
        return output_size*stride;  // inverse
//        return std::ceil((float)input_size/(float)stride);

    }else if(padding =="valid" || padding =="none"){
        return output_size*stride + ((kernel_size - 1) * dilation_rate);  // inverse
//        return std::ceil(((float)input_size - ((float)kernel_size - 1.0f) * (float)dilation_rate)/(float)stride);

    }else{
        cout<<padding<<endl;
        msg("Incorrect padding type", "ConvolDescriptorT::compute_output");
    }
    return -1;
}

int ConvolDescriptor::compute_output(vector<int> padding, int input_size, int kernel_size, int stride, int dilation_rate) {
    return (int)(((float)input_size - ((float)kernel_size - 1.0f) * (float)dilation_rate + (float)padding[0] + (float)padding[1] - 1.0f)/(float)stride + 1.0f);
}

vector<int> ConvolDescriptor::compute_padding(int output_size, int input_size, int kernel_size, int stride, string padding, bool row, int dilation_rate){
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
        int pad = (output_size-1) * stride + kernel_size - input_size + (dilation_rate-1) * (kernel_size-1);
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
