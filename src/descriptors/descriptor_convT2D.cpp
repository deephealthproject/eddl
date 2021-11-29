/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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

ConvolDescriptorT2D::ConvolDescriptorT2D() {}

ConvolDescriptorT2D::ConvolDescriptorT2D(int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding, const vector<int> &pads,
                                         int groups, const vector<int> &dilation_rate, bool use_bias, int mem){
    if (kernel_size.size() != 2) { msg("Kernels must have 3 dimensions", "ConvolDescriptorT2D::ConvolDescriptorT2D"); }
    if (strides.size() != 2) { msg("Strides must have 2 dimensions", "ConvolDescriptorT2D::ConvolDescriptorT2D"); }

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
        msg("Incorrect padding type (" + padding + ")", "ConvolDescriptorT2D::ConvolDescriptorT2D");
    }
}


ConvolDescriptorT2D::~ConvolDescriptorT2D(){
    // input, output, delta, params[], and gradients[], acc_gradients[] => deleted in ~Layer()
    if (O->isCPU()) {
        eddl_free(ptrI); // because get_fmem() now uses posix_memalign()
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

void ConvolDescriptorT2D::build(Tensor *A) {

    if (A->ndim != 4) msg("Tensors are not 4D", "ConvolDescriptorT2D::build");

    I = A;
    
    nk = A->shape[1]; //ksize[0];
    kr = ksize[1];
    kc = ksize[2];
    kz = A->shape[1]/groups;

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
        r = compute_output(pr, ir, kr, sr);

        vector<int>pc; pc.push_back(pads[2]);pc.push_back(pads[3]);
        c = compute_output(pc, ic, kc, sc);

    }else{  // Common padding (same/zeros)
        // Compute output
        z = nk;

        if (padding=="same,none") r = compute_output("same", ir, kr, sr);
        else if (padding=="none,same")  r = compute_output("none", ir, kr, sr);
        else r = compute_output(this->padding, ir, kr, sr);

        if (padding=="same,none") c = compute_output("none", ic, kc, sc);
        else if (padding=="none,same")  c = compute_output("same", ic, kc, sc);
        else c = compute_output(this->padding, ic, kc, sc);


        // Compute padding
        vector<int> padr = compute_padding(r, ir, kr, sr, this->padding,true);  // Order: [top, bottom]
        vector<int> padc = compute_padding(c, ic, kc, sc, this->padding,false);  // Order: [left, right]

        // Set padding
        this->pads.clear();
        this->pads = {padr[0], padr[1], padc[0], padc[1]};  // top, bottom, left, right
    }

#ifdef cCUDNN
       if(!A->isCPU()){
           if(pads[0] != pads[1] || pads[2] != pads[3]){
              msg("Warning: asymmetric padding not supported by cuDNN... fixing ... potential shapes mismatch later", "convt2d");
//             std::cout<<"Warning: asymmetric padding not supported by cuDNN... fixing ... potential shapes mismatch later"<<std::endl;
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
        msg("Invalid output shape", "ConvolDescriptorT2D::build");
    }

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);

    // Params
    K = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    bias = new Tensor(vector<int>{nk}, I->device);

    gK = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    gbias = new Tensor(vector<int>{nk}, I->device);

    if (I->isCPU()) {
        // mem for ptr, lowering im2col
        unsigned long int l_size =  (unsigned long)(A->shape[0] * r * c) * (unsigned long)(kr * kc * kz);
        ptrI=get_fmem(l_size,"ConvolDescriptorT::build");
        matI=Eigen::Map<Eigen::MatrixXf>(ptrI, r*c,kz*kr*kc);
	   _profile_add_tensor(A->shape[0] * r * c * kr * kc * kz);
    }
#ifdef cGPU
    else if (I->isGPU()) {
#ifndef cCUDNN
        if (mem_level>1) {
            // Lowering
            gpuIB=new Tensor(vector<int>{r*c,kc*kr*kz}, I->device);
        }
        else {
            // Big tensor with all the batch for lowering
            gpuIB=new Tensor(vector<int>{A->shape[0]*r*c,kc*kr*kz}, I->device);
            if (mem_level==0)
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

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    pads[0], pads[2],
                                    stride[0], stride[1],
                                    1,1,
                                    convolution_mode, data_type);
   cudnnSetConvolutionGroupCount(convolution_descriptor, groups);

   cudnnCreateTensorDescriptor(&xDesc);
   cudnnSetTensor4dDescriptor(xDesc, tensor_format, data_type,
                 in,iz,ir,ic);
   cudnnCreateFilterDescriptor(&wDesc);
   //CONVT we need to swap input channels with output so all other swappings (forward and backward functions) matches
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
        ptrI=get_fmem(A->shape[0] * r * c * kr * kc * kz,"ConvolDescriptorT::build");
    }
#endif
}

void ConvolDescriptorT2D::resize(int b)
{
    if (b==O->shape[0]) return;

    O->resize(b);


    // Prevent overflow. (512*512*512*3*3*3 = 3,623,878,656 > MAX_INT (2,147,483,647))
    unsigned long int l_size =  (unsigned long)(b * r * c) * (unsigned long)(kr * kc * kz);

    if (I->isCPU()) {
        eddl_free(ptrI); // because get_fmem() now uses posix_memalign()
        ptrI=get_fmem(l_size, "ConvolDescriptorT2D::build");
	   _profile_add_tensor(l_size);
    }
#ifdef cGPU
    else if (I->isGPU()) {
#ifndef cCUDNN
      if (mem_level<2)
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

   cudnn_env_init = -1;
   cudnn_conv_back_init = -1;

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
        ptrI=get_fmem(l_size, "ConvolDescriptorT::build");
    }
#endif


}

void ConvolDescriptorT2D::enable_distributed() {
    // Create and initialize the tensors for accumulating gradients in distributed training
    acc_gK = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    acc_gK->fill_(0.0);
    acc_gbias = new Tensor(vector<int>{nk}, I->device);
    acc_gbias->fill_(0.0);
}

int ConvolDescriptorT2D::compute_output(const string& padding, int input_size, int kernel_size, int stride, int dilation_rate){
    if (padding=="same" || padding =="zeros") {
        return input_size*stride;  // inverse
        //return std::ceil((float)input_size/(float)stride);

    }else if(padding =="valid" || padding =="none"){
        return input_size*stride + ((kernel_size - 1) * dilation_rate);  // inverse
        //return std::ceil(((float)input_size - ((float)kernel_size - 1.0f) * (float)dilation_rate)/(float)stride);

    }else{
      cout<<padding<<endl;
        msg("Incorrect padding type", "ConvolDescriptorT2D::compute_output");
    }
    return -1;
}

int ConvolDescriptorT2D::compute_output(vector<int> padding, int input_size, int kernel_size, int stride, int dilation_rate) {
    return  stride * (input_size - 1) + ((kernel_size - 1) * dilation_rate + 1) - padding[0] - padding[1];  // inverse
    //return (int)(((float)input_size - ((float)kernel_size - 1.0f) * (float)dilation_rate + (float)padding[0] + (float)padding[1] - 1.0f)/(float)stride + 1.0f);
}

vector<int> ConvolDescriptorT2D::compute_padding(int output_size, int input_size, int kernel_size, int stride, string padding, bool row){
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
        int pad = (input_size-1) * stride + kernel_size - output_size;  // Inverse
        //int pad = (output_size-1) * stride + kernel_size - input_size;
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
        msg("Incorrect padding type", "ConvolDescriptorT2D::compute_padding");
    }

    return {-1};
}
