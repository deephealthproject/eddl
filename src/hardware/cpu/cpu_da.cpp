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
    Tensor *B = A->clone();
    for(int i=0; i<A->shape[0];i++) {
        for(int j=0; j<A->shape[1];j++) {

            vector<int> pos = {i - shift[0], j - shift[1]};
            if (B->valid_indices(pos)){
                A->set_({i, j}, B->get_(pos));
            }else{

                // TODO: SUPER INEFFICIENT!!! (TEMP)
                if(mode=="constant"){
                    A->set_({i, j}, constant);
                }

            }

        }
    }
}

void cpu_rotate_(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){

}

void cpu_scale_(Tensor *A, float factor, bool reshape, string mode, float constant){

}

void cpu_flip_(Tensor *A, int axis){

}

void cpu_crop_(Tensor *A, vector<int> coords_from, vector<int> coords_to){

}

void cpu_cutout_(Tensor *A, vector<int> coords_from, vector<int> coords_to){

}