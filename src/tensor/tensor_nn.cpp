#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

#include "tensor.h"
#include "../utils.h"

#include "../hardware/cpu/cpu_convol.h"


#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;

////////////////////////////////
//// CONVOLUTIONS
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

void ConvolDescriptor::resize(Tensor *A)
{
    I=A;

    delete O;
    delete D;
    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);

    if (I->isCPU()) {
        delete ptrI;
        ptrI=get_fmem(A->shape[0] * r * c * kr * kc * kz,"ConvolDescriptor::build");
    }
#ifdef cGPU
    else if (I->isGPU()) {
      delete gpuIB;
      gpuIB=new Tensor(vector<int>{A->shape[0]*r*c,kc*kr*kz}, I->device);
    }
#endif

}

/////////////////////////////////////////////////////////////////////
//// Conv2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D(ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    D->O->tsem->lock();
    if (D->I->isCPU()) {
        cpu_conv2D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         //gpu_conv2D_old(D);
         gpu_conv2D(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->O->tsem->unlock();
}

/////////////////////////////////////////////////////////////////////
//// Conv2D Grad
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D_grad(ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    D->gK->tsem->lock();
    if (D->I->isCPU()) {
        D->gK->set(0.0);
        D->gbias->set(0.0);
        cpu_conv2D_grad(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         D->gK->set(0.0);
         D->gbias->set(0.0);
         gpu_conv2D_grad(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->gK->tsem->unlock();
}

/////////////////////////////////////////////////////////////////////
//// Conv2D Back
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D_back(ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    D->ID->tsem->lock();
    if (D->I->isCPU()) {
        cpu_conv2D_back(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         gpu_conv2D_back(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->ID->tsem->unlock();
}


////////////////////////////////
////  POOLING

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st,
                               const vector<int> &p) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());

    if (ksize.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (stride.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (pad.size() != 2) msg("Padding must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
}

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st, string p) {
    if (ks.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (st.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");

    ksize = ks;
    stride = st;

    if (p == "same") {
        pad.push_back(ksize[0] / 2);
        pad.push_back(ksize[1] / 2);
    } else if (p == "none") {
        pad.push_back(0);
        pad.push_back(0);
    } else msg("Incorrect padding type", "PoolDescriptor::PoolDescriptor");
}


void PoolDescriptor::build(Tensor *A) {
    if (A->ndim != 4) msg("Tensors are not 4D", "PoolDescriptor::build");

    I = A;

    kr = ksize[0];
    kc = ksize[1];

    sr = stride[0];
    sc = stride[1];

    iz = A->shape[1];
    ir = A->shape[2];
    ic = A->shape[3];

    padr = pad[0];
    padc = pad[1];

    z = iz;
    r = (ir - kr + 2 * padr) / sr + 1;
    //if (kr%2==0) r--;
    c = (ic - kc + 2 * padc) / sc + 1;
    //if (kc%2==0) c--;

    if ((r <= 0) || (c <= 0))
        msg("Invalid output shape", "PoolDescriptor::build");

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);


}

void PoolDescriptor::resize(Tensor *A) {

    I = A;

    delete O;
    delete D;

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);
}

/////////////////////////////////////////////////////////////////////
//// MPool2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::MPool2D(PoolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::MPool2D");

    D->O->tsem->lock();
    if (D->I->isCPU()) {
        cpu_mpool2D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
        gpu_mpool2D(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->O->tsem->unlock();
}

/////////////////////////////////////////////////////////////////////
//// MPool2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::MPool2D_back(PoolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::MPool2D_back");

    D->ID->tsem->lock();
    if (D->I->isCPU()) {

        cpu_mpool2D_back(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
        gpu_mpool2D_back(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->ID->tsem->unlock();
}
