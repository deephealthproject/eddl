/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "descriptors.h"

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st,
                               const vector<int> &p, int mem) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());
    mem_level=mem;

    if (ksize.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (stride.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    //if (pad.size() != 2) msg("Padding must have 2 dimensions", "PoolDescriptor::PoolDescriptor");

}

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st, string p, int mem) {
    if (ks.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (st.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");

    ksize = ks;
    stride = st;
    mem_level=mem;

    if (p == "same") {

      pad.push_back(ksize[0] / 2);
      pad.push_back(ksize[0] / 2);
      if (ksize[0]%2==0) pad[1]--;

      pad.push_back(ksize[1] / 2);
      pad.push_back(ksize[1] / 2);
      if (ksize[1]%2==0) pad[3]--;

    } else if (p == "none") {
        pad.push_back(0);
        pad.push_back(0);
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


    padrt = pad[0];
    padrb = pad[1];
    padcl = pad[2];
    padcr = pad[3];



    z = iz;
    r = (ir - kr + padrt + padrb) / sr + 1;
    c = (ic - kc + padcl + padcr) / sc + 1;

    if ((r <= 0) || (c <= 0))
        msg("Invalid output shape", "PoolDescriptor::build");

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    if (mem_level<2) D = new Tensor(O->getShape(), A->device);

    size=0;
    for(int k=0;k<iz;k++)
      for(int i=-padrt;i<=ir+padrb-kr;i+=sr)
        for(int j=-padcl;j<=ic+padcr-kc;j+=sc,size++) {}


}

void PoolDescriptor::resize(int b) {
  if (b==O->shape[0]) return;

  O->resize(b);
  if (mem_level<2) D->resize(b);

}
