/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <utility>

#include "cpu_hw.h"

// CPU: Data augmentation (in-place) ********************************************
Tensor* cpu_shift(Tensor *A, vector<int> shift, string mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
    Tensor *B = Tensor::full(A->getShape(), constant);

    for(int i=0; i<B->shape[0];i++) {
        for(int j=0; j<B->shape[1];j++) {

            vector<int> pos = {i - shift[0], j - shift[1]};
            if (A->valid_indices(pos)){
                B->set_({i, j}, A->get_(pos));
            }

        }
    }

    return B;
}

Tensor* cpu_rotate(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    return A;
}

Tensor* cpu_scale(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    Tensor *B;
    vector<int>offsets(A->ndim, 0);

    // Resize keeping the original size (e.g.: if zoom-out, add zeros, else "crop")
    if(reshape) { B = Tensor::full(new_shape, constant);
    } else {
        B = Tensor::full(A->getShape(), constant);

        // Compute offset to center the inner matrix (zoom-out)
        for(int i=0; i<offsets.size(); i++){
            offsets[i] = A->shape[i]/2.0f - new_shape[i]/2.0f;
        }
    }

    for(int i=0; i<B->shape[0];i++) {
        for(int j=0; j<B->shape[1];j++) {
            // Interpolate indices
            int Ai = (i * A->shape[0]) / new_shape[0];
            int Aj = (j * A->shape[1]) / new_shape[1];

            vector<int> pos = {Ai, Aj};
            if (A->valid_indices(pos)){
                B->set_({i + offsets[0] , j + offsets[1]}, A->get_(pos));
            }

        }
    }

    return B;
}

Tensor* cpu_flip(Tensor *A, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Tensor *B = A->clone();

    for(int i=0; i<B->shape[0];i++) {
        for(int j=0; j<B->shape[1];j++) {

            vector<int> pos = {i, j};
            pos[axis] = (B->shape[axis]-1) - pos[axis];
            if (A->valid_indices(pos)){
                B->set_({i, j}, A->get_(pos));
            }

        }
    }

    return B;
}

Tensor* cpu_crop(Tensor *A, vector<int> coords_from, vector<int> coords_to, bool reshape, float constant){
    Tensor *B;
    vector<int> new_shape;

    // If "True", return a smaller tensor. Else, fill the non-cropped region
    if(reshape) {
        for(int i=0; i<A->ndim; i++){
            new_shape.push_back(coords_to[i] - coords_from[i] + 1);
        }
    } else { new_shape = A->shape; }

    B = Tensor::full(new_shape, constant);
    for(int Ai=coords_from[0], Bi=0; Ai<=coords_to[0]; Ai++, Bi++) {
        for(int Aj=coords_from[1], Bj=0; Aj<=coords_to[1]; Aj++, Bj++) {

            if (A->valid_indices({Ai, Aj})){
                B->set_({Bi, Bj}, A->get_({Ai, Aj}));
            }

        }
    }

    return B;
}

Tensor* cpu_cutout(Tensor *A, vector<int> coords_from, vector<int> coords_to, float constant){
    Tensor *B = A->clone();

    for(int i=coords_from[0]; i<=coords_to[0];i++) {
        for(int j=coords_from[1]; j<=coords_to[1];j++) {

            // Fill values in region
            if (B->valid_indices({i, j})){
                B->set_({i, j}, constant);
            }

        }
    }

    return B;
}