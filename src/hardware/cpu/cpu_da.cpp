/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <iostream>
#include <utility>

#include "cpu_hw.h"

// CPU: Data augmentation (in-place) ********************************************
Tensor* cpu_shift(Tensor *A, vector<int> shift, string mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
    Tensor *B = Tensor::full(A->getShape(), constant);

    for(int i=0; i< B->size; i++){
        // Get indices
        vector<int> B_pos = B->get_indices_rowmajor(i);
        vector<int> A_pos(B_pos);

        // Compute positions
        for(int j=0; j<shift.size(); j++){
            A_pos[A_pos.size()-shift.size()+j] -= shift[j];
        }

        // Change values
        if (A->valid_indices(A_pos)){
            B->set_(B_pos, A->get_(A_pos));
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
    vector<int> A_pos(A->ndim, 0);
    vector<int> B_pos;
    int *offsets = new int[A->ndim];

    // Resize keeping the original size (e.g.: if zoom-out, add zeros, else "crop")
    if(reshape) {
        B = Tensor::full(new_shape, constant);
    } else {
        B = Tensor::full(A->getShape(), constant);

        // Compute offset to center the inner matrix (zoom-out)
        for(int i=0; i<A->ndim; i++){
            offsets[i] = A->shape[i]/2.0f - new_shape[i]/2.0f;
        }
    }

    for(int i=0; i< B->size; i++){
        // Compute interpolated indices
        B_pos = B->get_indices_rowmajor(i);
        for(int j=0; j<A->ndim; j++){
            A_pos[j] = (B_pos[j] * A->shape[j]) / new_shape[j];
            B_pos[j] += offsets[j];
        }

        // Change values
        if (A->valid_indices(A_pos)){
            B->set_(B_pos, A->get_(A_pos));
        }
    }

    return B;
}

Tensor* cpu_flip(Tensor *A, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Tensor *B = A->clone();

    for(int i=0; i< B->size; i++) {
        // Compute interpolated indices
        vector<int> B_pos = B->get_indices_rowmajor(i);
        vector<int> A_pos(B_pos);

        A_pos[axis] = (B->shape[axis]-1) - A_pos[axis];
        if (A->valid_indices(A_pos)){
            B->set_(B_pos, A->get_(A_pos));
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