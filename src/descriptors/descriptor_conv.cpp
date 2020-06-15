/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
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
#include "eddl/hardware/gpu/nn/gpu_nn.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#endif

ConvolDescriptor::ConvolDescriptor() {}

ConvolDescriptor::ConvolDescriptor(const vector<int> &ks, const vector<int> &st, const vector<int> &p, int mem) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());
    mem_level=mem;

    this->padding = "custom";

    if (ksize.size() != 3) msg("Kernels must have 3 dimensions", "ConvolDescriptor::ConvolDescriptor");
    if (stride.size() != 2) msg("Strides must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor");
}

ConvolDescriptor::ConvolDescriptor(int filters, const vector<int> &ks, const vector<int> &st, const string& p, bool ub, int mem) {
    if (ks.size() != 2) { msg("Kernels must have 3 dimensions", "ConvolDescriptor::ConvolDescriptor"); }
    if (st.size() != 2) { msg("Strides must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor"); }

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
        msg("Incorrect padding type", "ConvolDescriptor::ConvolDescriptor");
    }

}


void ConvolDescriptor::build(Tensor *A) {

    if (A->ndim != 4) msg("Tensors are not 4D", "ConvolDescriptor::build");

    I = A;

    nk = ksize[0];
    kr = ksize[1];
    kc = ksize[2];
    kz = A->shape[1];

    sr = stride[0];
    sc = stride[1];

    iz = A->shape[1];
    ir = A->shape[2];
    ic = A->shape[3];

    if(this->padding=="custom"){  // Known padding
        // Compute output
        z = nk;
        vector<int>pr; pr.push_back(pad[0]);pr.push_back(pad[1]);
        r = compute_output(pr, ir, kr, sr);

        vector<int>pc; pc.push_back(pad[2]);pc.push_back(pad[3]);
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
        pad = {padr[0], padr[1], padc[0], padc[1]};  // top, bottom, left, right
    }

    padrt = pad[0]; padrb = pad[1];  // rows: top-bottom
    padcl = pad[2]; padcr = pad[3];  // cols: left-right

    if ((r <= 0) || (c <= 0)) {
        cout<<"rows="<<r<<" cols"<<c<<endl;
        msg("Invalid output shape", "ConvolDescriptor::build");
    }

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
//    if (!mem_level) { D = new Tensor(O->shape, A->device); }

    // Params
    K = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    bias = new Tensor(vector<int>{nk}, I->device);

    gK = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    gbias = new Tensor(vector<int>{nk}, I->device);

    if (I->isCPU()) {
        // mem for ptr, lowering im2col
        ptrI=get_fmem(A->shape[0] * r * c * kr * kc * kz,"ConvolDescriptor::build");
	 _profile_add_tensor(A->shape[0] * r * c * kr * kc * kz);
        new(&matK) Eigen::Map<Eigen::MatrixXf>(K->ptr, kr * kc * kz, nk);
        new(&matgK) Eigen::Map<Eigen::MatrixXf>(gK->ptr, kr * kc * kz, nk);
        // convolution: matC=matA*matK
    }
#ifdef cGPU
    else if (I->isGPU()) {

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
#endif

#ifdef cFPGA
    if (I->isFPGA()) {
	// We allocate memory on the FGPA for the im2col buffer
	fpga_sizeI = A->shape[0] * r * c * kr * kc * kz * sizeof(float);
	fpga_ptrI = fpga_create_memory(fpga_sizeI);
	// We allocate also on cpu so to ease the cpuemu flow
        // mem for ptr, lowering im2col
        ptrI=get_fmem(A->shape[0] * r * c * kr * kc * kz,"ConvolDescriptor::build");
        new(&matK) Eigen::Map<Eigen::MatrixXf>(K->ptr, kr * kc * kz, nk);
        new(&matgK) Eigen::Map<Eigen::MatrixXf>(gK->ptr, kr * kc * kz, nk);
    }
#endif
}

void ConvolDescriptor::resize(int b)
{
    if (b==O->shape[0]) return;

    O->resize(b);
//    if (!mem_level) D->resize(b);

    if (I->isCPU()) {
        delete ptrI;
        ptrI=get_fmem(b * r * c * kr * kc * kz, "ConvolDescriptor::build");
	 _profile_add_tensor(b * r * c * kr * kc * kz);
    }
#ifdef cGPU
    else if (I->isGPU()) {
        if (mem_level<2)
            gpuIB->resize(b*r*c);
        if (mem_level==0) {
            delete gpuOB;
            gpuOB=new Tensor(vector<int>{z,b*r*c}, I->device);
        }
    }
#endif

#ifdef cFPGA
    else if (I->isFPGA()) {
        // We reallocate memory on the FGPA for the im2col buffer
	fpga_destroy_memory(fpga_ptrI);
	fpga_sizeI = b * r * c * kr * kc * kz * sizeof(float);
        fpga_ptrI = fpga_create_memory(fpga_sizeI);
        // We do the same on the CPU side (for smooth cpuemu)
	delete ptrI;
        ptrI=get_fmem(b * r * c * kr * kc * kz, "ConvolDescriptor::build");
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

int ConvolDescriptor::compute_output(const string& padding, int input_size, int kerkel_size, int stride, int dilation_rate){
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

int ConvolDescriptor::compute_output(vector<int> padding, int input_size, int kerkel_size, int stride, int dilation_rate) {
    return (int)(((float)input_size - ((float)kerkel_size - 1.0f) * (float)dilation_rate + (float)padding[0] + (float)padding[1] - 1.0f)/(float)stride + 1.0f);
}

vector<int> ConvolDescriptor::compute_padding(int output_size, int input_size, int kerkel_size, int stride, string padding, bool row){
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
