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

// CPU: Data augmentation (2D Optimized) ********************************************
Tensor* cpu_shift(Tensor *A, vector<int> shift, string mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
    Tensor *B = Tensor::full(A->getShape(), constant);

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {
                    int Ai = Bi - shift[0];
                    int Aj = Bj - shift[1];

                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                        int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                        int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    }

                }
            }

        }
    }

    return B;
}

Tensor* cpu_rotate(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    // TODO: IMPLEMENT
    return A;
}

Tensor* cpu_scale(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    Tensor *B;
    int offsets[2] = {0, 0};

    // Resize keeping the original size (e.g.: if zoom-out, add zeros, else "crop")
    if(reshape) { B = Tensor::full(new_shape, constant);
    } else {
        B = Tensor::full(A->getShape(), constant);

        // Compute offset to center the inner matrix (zoom-out)
        offsets[0] = A->shape[0]/2.0f - new_shape[2]/2.0f;
        offsets[1] = A->shape[1]/2.0f - new_shape[3]/2.0f;
    }

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {
                    // Interpolate indices
                    int Ai = (Bi * A->shape[2]) / new_shape[2];
                    int Aj = (Bj * A->shape[3]) / new_shape[3];

                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                        int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                        int B_pos = b*B->stride[0] + c*B->stride[1] + (Bi+ offsets[0])*B->stride[2] + (Bj+ offsets[1])*B->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    }

                }
            }

        }
    }

    return B;
}

Tensor* cpu_flip(Tensor *A, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Tensor *B = A->clone();

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {
                    int pos[2] = {Bi, Bj}; pos[axis] = (B->shape[axis+2]-1) - pos[axis];
                    int Ai = pos[0]; int Aj = pos[1];

                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                        int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                        int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    }

                }
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

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Ai=coords_from[0], Bi=0; Ai<=coords_to[0]; Ai++, Bi++) {
                for(int Aj=coords_from[1], Bj=0; Aj<=coords_to[1]; Aj++, Bj++) {

                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                        int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                        int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    }

                }
            }
        }
    }

    return B;
}

Tensor* cpu_cutout(Tensor *A, vector<int> coords_from, vector<int> coords_to, float constant){
    Tensor *B = A->clone();

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Bi=coords_from[0]; Bi<=coords_to[0];Bi++) {
                for(int Bj=coords_from[1]; Bj<=coords_to[1];Bj++) {

                    if (Bi >= 0 && Bi < A->shape[2] && Bj >= 0 && Bj < A->shape[3]){
                        int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                        B->ptr[B_pos] = constant;
                    }

                }
            }
        }
    }

    return B;
}


// CPU: Data augmentation (2D No-Optimized)[Temp] ********************************************
Tensor* cpu_shift_no(Tensor *A, vector<int> shift, string mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
    Tensor *B = Tensor::full(A->getShape(), constant);

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int i=0; i<B->shape[2];i++) {
                for(int j=0; j<B->shape[3];j++) {

                    vector<int> pos = {b, c, i - shift[0], j - shift[1]};
                    if (A->valid_indices(pos)){
                        B->set_({b, c, i, j}, A->get_(pos));
                    }

                }
            }

        }
    }

    return B;
}

Tensor* cpu_rotate_no(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    // TODO: IMPLEMENT
    return A;
}

Tensor* cpu_scale_no(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    Tensor *B;
    int offsets[2] = {0, 0};

    // Resize keeping the original size (e.g.: if zoom-out, add zeros, else "crop")
    if(reshape) { B = Tensor::full(new_shape, constant);
    } else {
        B = Tensor::full(A->getShape(), constant);

        // Compute offset to center the inner matrix (zoom-out)
        offsets[0] = A->shape[0]/2.0f - new_shape[2]/2.0f;
        offsets[1] = A->shape[1]/2.0f - new_shape[3]/2.0f;
    }

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int i=0; i<B->shape[2];i++) {
                for(int j=0; j<B->shape[3];j++) {
                    // Interpolate indices
                    int Ai = (i * A->shape[2]) / new_shape[2];
                    int Aj = (j * A->shape[3]) / new_shape[3];

                    vector<int> pos = {b, c, Ai, Aj};
                    if (A->valid_indices(pos)){
                        B->set_({b, c, i + offsets[0] , j + offsets[1]}, A->get_(pos));
                    }

                }
            }

        }
    }

    return B;
}

Tensor* cpu_flip_no(Tensor *A, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Tensor *B = A->clone();
    axis += 2;

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int i=0; i<B->shape[2];i++) {
                for(int j=0; j<B->shape[3];j++) {

                    vector<int> pos = {b, c, i, j};
                    pos[axis] = (B->shape[axis]-1) - pos[axis];
                    if (A->valid_indices(pos)){
                        B->set_({b, c, i, j}, A->get_(pos));
                    }

                }
            }

        }
    }

    return B;
}

Tensor* cpu_crop_no(Tensor *A, vector<int> coords_from, vector<int> coords_to, bool reshape, float constant){
    Tensor *B;
    vector<int> new_shape;

    // If "True", return a smaller tensor. Else, fill the non-cropped region
    if(reshape) {
        for(int i=0; i<A->ndim; i++){
            new_shape.push_back(coords_to[i] - coords_from[i] + 1);
        }
    } else { new_shape = A->shape; }

    B = Tensor::full(new_shape, constant);

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Ai=coords_from[0], Bi=0; Ai<=coords_to[0]; Ai++, Bi++) {
                for(int Aj=coords_from[1], Bj=0; Aj<=coords_to[1]; Aj++, Bj++) {

                    if (A->valid_indices({b, c, Ai, Aj})){
                        B->set_({b, c, Bi, Bj}, A->get_({b, c, Ai, Aj}));
                    }

                }
            }
        }
    }

    return B;
}

Tensor* cpu_cutout_no(Tensor *A, vector<int> coords_from, vector<int> coords_to, float constant){
    Tensor *B = A->clone();

    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int i=coords_from[0]; i<=coords_to[0];i++) {
                for(int j=coords_from[1]; j<=coords_to[1];j++) {

                    // Fill values in region
                    if (B->valid_indices({b, c, i, j})){
                        B->set_({b, c, i, j}, constant);
                    }

                }
            }
        }
    }

    return B;
}


// CPU: Data augmentation (Generic)[Temp] ********************************************
Tensor* cpu_shift_gen(Tensor *A, vector<int> shift, string mode, float constant) {
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

Tensor* cpu_rotate_gen(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    // TODO: IMPLEMENT
    return A;
}

Tensor* cpu_scale_gen(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant){
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

Tensor* cpu_flip_gen(Tensor *A, int axis){
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

Tensor* cpu_crop_gen(Tensor *A, vector<int> coords_from, vector<int> coords_to, bool reshape, float constant){
    Tensor *B;
    vector<int> new_shape;

    // If "True", return a smaller tensor. Else, fill the non-cropped region
    if(reshape) {
        for(int i=0; i<A->ndim; i++){
            new_shape.push_back(coords_to[i] - coords_from[i] + 1);
        }
    } else { new_shape = A->shape; }

    B = Tensor::full(new_shape, constant);

    // TODO: Implement

    return B;
}

Tensor* cpu_cutout_gen(Tensor *A, vector<int> coords_from, vector<int> coords_to, float constant){
    Tensor *B = A->clone();

    // TODO: Implement

    return B;
}