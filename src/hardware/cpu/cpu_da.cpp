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
void cpu_shift_(Tensor *A, vector<int> shift, bool reshape, string mode, float constant) {
    Tensor *B = Tensor::full(A->getShape(), constant);

    for(int i=0; i<B->shape[0];i++) {
        for(int j=0; j<B->shape[1];j++) {

            vector<int> pos = {i - shift[0], j - shift[1]};
            if (A->valid_indices(pos)){
                B->set_({i, j}, A->get_(pos));
            }else{}

        }
    }

    *A = *B;
}

void cpu_rotate_(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){

}

Tensor* cpu_scale(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant){
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
            }else{}

        }
    }

    return B;
}

void cpu_flip_(Tensor *A, int axis){
    Tensor *B = A->clone();

    for(int i=0; i<B->shape[0];i++) {
        for(int j=0; j<B->shape[1];j++) {

            vector<int> pos = {i, j};
            pos[axis] = (B->shape[axis]-1) - pos[axis];
            if (A->valid_indices(pos)){
                B->set_({i, j}, A->get_(pos));
            }else{}

        }
    }

    *A = *B;
}

void cpu_crop_(Tensor *A, vector<int> coords_from, vector<int> coords_to){

}

void cpu_cutout_(Tensor *A, vector<int> coords_from, vector<int> coords_to){

}