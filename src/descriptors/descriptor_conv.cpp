/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "descriptors.h"


#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#include "../hardware/gpu/gpu_hw.h"
#include "../hardware/gpu/nn/gpu_nn.h"
#endif

ConvolDescriptor::ConvolDescriptor() {}

ConvolDescriptor::ConvolDescriptor(int filters, const vector<int> &ks, const vector<int> &st, string p) {
    if (ks.size() != 2) { msg("Kernels must have 3 dimensions", "ConvolDescriptor::ConvolDescriptor"); }
    if (st.size() != 2) { msg("Strides must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor"); }

    // Add filters to kernel_size
    ksize = vector<int>(ks);
    ksize.insert(ksize.begin(), 1, filters);
    stride = vector<int>(st.begin(), st.end());

    if (p == "same") {
        pad.push_back(ksize[1] / 2);
        pad.push_back(ksize[2] / 2);
    } else if (p == "none") {
        pad.push_back(0);
        pad.push_back(0);
    } else msg("Incorrect padding type", "ConvolDescriptor::ConvolDescriptor");

}

ConvolDescriptor::ConvolDescriptor(const vector<int> &ks, const vector<int> &st,
                                   const vector<int> &p) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());

    if (ksize.size() != 3) msg("Kernels must have 3 dimensions", "ConvolDescriptor::ConvolDescriptor");
    if (stride.size() != 2) msg("Strides must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor");
    if (pad.size() != 2) msg("Padding must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor");
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

    padr = pad[0];
    padc = pad[1];

    z = nk;
    r = (ir - kr + 2 * padr) / sr + 1;
    c = (ic - kc + 2 * padc) / sc + 1;

    if ((r <= 0) || (c <= 0))
        msg("Invalid output shape", "ConvolDescriptor::build");

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);

    // Params
    K = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    bias = new Tensor(vector<int>{nk}, I->device);

    gK = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    gbias = new Tensor(vector<int>{nk}, I->device);

    if (I->isCPU()) {
        // mem for ptr, lowering im2col
        ptrI=get_fmem(A->shape[0] * r * c * kr * kc * kz,"ConvolDescriptor::build");
        new(&matK) Eigen::Map<Eigen::MatrixXf>(K->ptr, kr * kc * kz, nk);
        new(&matgK) Eigen::Map<Eigen::MatrixXf>(gK->ptr, kr * kc * kz, nk);
        // convolution: matC=matA*matK
    }
#ifdef cGPU
    else if (I->isGPU()) {
      // Big tensor with all the lowering
      gpuIB=new Tensor(vector<int>{A->shape[0]*r*c,kc*kr*kz}, I->device);

      // Tensor with variable shared ptr, delete create ptr
      gpuI=new Tensor(vector<int>{r*c,kc*kr*kz}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpuI->ptr);

      gpuO=new Tensor(vector<int>{z,r*c}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpuO->ptr);
      gpuD=new Tensor(vector<int>{z,r*c}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpuD->ptr);

      gpuK=new Tensor(vector<int>{z,kc*kr*kz}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpuK->ptr);
      gpugK=new Tensor(vector<int>{z,kc*kr*kz}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpugK->ptr);
    }
#endif
}

void ConvolDescriptor::resize(int b)
{
    if (b==O->shape[0]) return;

    O->resize(b);
    D->resize(b);

    if (I->isCPU()) {
        delete ptrI;
        ptrI=get_fmem(b * r * c * kr * kc * kz,"ConvolDescriptor::build");
    }
#ifdef cGPU
    else if (I->isGPU()) {
      gpuIB->resize(b*r*c);
    }
#endif

}





////
