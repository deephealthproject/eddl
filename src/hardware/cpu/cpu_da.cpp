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
#include <cmath>

#include "cpu_hw.h"
#include "../../random.h"


// CPU: Data augmentation (2D Optimized) ********************************************
void cpu_shift(Tensor *A, Tensor *B, vector<int> shift, int mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html

    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        for(int c=0; c<B->shape[1]; c++) {
            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {

                    int Ai = Bi - shift[0];
                    int Aj = Bj - shift[1];

                    int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                        int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    }else{
                        if(mode==0){ // constant
                            B->ptr[B_pos] = constant;
                        }
                    }

                }
            }
        }

    }
}

void cpu_rotate(Tensor *A, Tensor *B, float angle, vector<int> axis, int mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
}

void cpu_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    // I use "new_shape" because I might want to keep the shape of B, but thinking of it as a bigger/smaller matrix
    // If the new_shape is smaller than B, performs a downscale with padding

    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        for(int c=0; c<B->shape[1]; c++) {
            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {

                    // Interpolate indices
                    int Ai = (Bi * A->shape[2]) / new_shape[0];
                    int Aj = (Bj * A->shape[3]) / new_shape[1];

                    int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                        int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    }else{
                        if(mode==0){ // constant
                            B->ptr[B_pos] = constant;
                        }
                    }

                }
            }
        }

    }
}

void cpu_flip(Tensor *A, Tensor *B, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    #pragma omp parallel for
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
}

void cpu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant){
    int offsets[2] = {0, 0};
    // Center crop (if the if the crop is smaller than B)
    offsets[0] = (B->shape[2] - (coords_to[0]-coords_from[0]+1))/2.0f;
    offsets[1] = (B->shape[3] - (coords_to[1]-coords_from[1]+1))/2.0f;

    //#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        for(int c=0; c<B->shape[1]; c++) {
            for(int Ai=coords_from[0], Bi=0; Ai<=coords_to[0]; Ai++, Bi++) {
                for(int Aj=coords_from[1], Bj=0; Aj<=coords_to[1]; Aj++, Bj++) {

                    int B_pos = b*B->stride[0] + c*B->stride[1] + (Bi+offsets[0])*B->stride[2] + (Bj+offsets[1])*B->stride[3];
                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                        int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    }else{
                        B->ptr[B_pos] = constant;
                    }

                }
            }
        }

    }
}


void cpu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant){
    int A_hc = coords_to[0]-coords_from[0]+1;
    int A_wc = coords_to[1]-coords_from[1]+1;

    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        for(int c=0; c<B->shape[1]; c++) {
            for(int Bi=0; Bi<B->shape[2]; Bi++) {
                for(int Bj=0; Bj<B->shape[3]; Bj++) {

                    // Interpolate indices
                    int Ai = (Bi * A_hc) / B->shape[2] + coords_from[0];
                    int Aj = (Bj * A_wc) / B->shape[3] + coords_from[1];

                    int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                    int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];

                    B->ptr[B_pos] = A->ptr[A_pos];
                }
            }
        }

    }
}

void cpu_cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant){

    #pragma omp parallel for
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
}


// CPU: Data augmentation (2D Optimized) ********************************************
void cpu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html

    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        int shift_x = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));
        int shift_y = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));

        for(int c=0; c<B->shape[1]; c++) {
            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {
                    int Ai = Bi - shift_y;
                    int Aj = Bj - shift_x;

                    int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                        int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    }else{
                        if(mode==0){ // constant
                            B->ptr[B_pos] = constant;
                        }
                    }

                }
            }
        }

    }
}

void cpu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> axis, int mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
}

void cpu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    // I use "new_shape" because I might want to keep the shape of B, but thinking of it as a bigger/smaller matrix
    // If the factor is less than 1.0f, performs a downscale with padding
    int offsets[2] = {0, 0};
    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        // TODO: Center image
        float scale = uniform(factor[0], factor[1]);
        int new_shape_x = (int)(A->shape[3] * scale);
        int new_shape_y = (int)(A->shape[2] * scale);


        // Center crop (if the if the crop is smaller than B)
        offsets[0] = (A->shape[2] - new_shape_y)/2.0f;
        offsets[1] = (A->shape[3] - new_shape_x)/2.0f;

        for(int c=0; c<B->shape[1]; c++) {
            for(int Ai=0; Ai<A->shape[2];Ai++) {
                for(int Aj=0; Aj<A->shape[3];Aj++) {
                    // Interpolate indices
//                    // TODO: FIX
//                    int Bi_big = (Ai * new_shape_y) / A->shape[2];
//                    int Bj_big = (Aj * new_shape_x) / A->shape[3];
//                    int Bi = Bi_big + offsets[0];
//                    int Bj = Bj_big + offsets[1];
//
//                    int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
//                    if (Bi_big >= offsets[0] && Bi_big < B->shape[2] && Bj_big >= 0 && Bj < B->shape[3]){
//                        B->ptr[A_pos] = A->ptr[A_pos];
//                    }else{
//                        if(mode==0){ // constant
//                            B->ptr[A_pos] = constant;
//                        }
//                    }

                }
            }
        }

    }
}

void cpu_flip_random(Tensor *A, Tensor *B, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        bool apply = uniform(0.0f, 1.0f) >= 0.5f;


        for(int c=0; c<B->shape[1]; c++) {
            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {

                    if (apply){
                        // Apply randomly
                        int pos[2] = {Bi, Bj}; pos[axis] = (B->shape[axis+2]-1) - pos[axis];
                        int Ai = pos[0]; int Aj = pos[1];

                        if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                            int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                            int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                            B->ptr[B_pos] = A->ptr[A_pos];
                        }
                    }else{
                        int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                        B->ptr[B_pos] = A->ptr[B_pos];
                    }

                }
            }
        }

    }
}


void cpu_crop_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    // Performs a crop with padding (Keeps the original size)
    int offsets[2] = {0, 0};

    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        // Compute random coordinates
        int x1 = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));
        int x2 = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));
        int y1 = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));
        int y2 = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));

        int coords_from_x = std::min(x1, x2);
        int coords_to_x = std::max(x1, x2);
        int coords_from_y = std::min(y1, y2);
        int coords_to_y = std::max(y1, y2);

//        // Center crop (if the if the crop is smaller than B)
//        offsets[0] = (B->shape[2] - (coords_to_y-coords_from_y+1))/2.0f;
//        offsets[1] = (B->shape[3] - (coords_to_x-coords_from_x+1))/2.0f;

        for(int c=0; c<B->shape[1]; c++) {
            for(int Bi=0; Bi<B->shape[2]; Bi++) {
                for(int Bj=0; Bj<B->shape[3]; Bj++) {

                    int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                    if (Bi >= coords_from_y && Bi <= coords_to_y && Bj >= coords_from_x && Bj <= coords_to_x){
                        B->ptr[B_pos] = A->ptr[B_pos];
                    }else{
                        B->ptr[B_pos] = constant;
                    }

                }
            }
        }

    }
}

void cpu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        // Compute random coordinates
        int x1 = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));
        int x2 = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));
        int y1 = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));
        int y2 = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));

        int coords_from_x = std::min(x1, x2);
        int coords_to_x = std::max(x1, x2);
        int coords_from_y = std::min(y1, y2);
        int coords_to_y = std::max(y1, y2);

        int A_wc = coords_to_x-coords_from_x+1;
        int A_hc = coords_to_y-coords_from_y+1;

        for(int c=0; c<B->shape[1]; c++) {
            for(int Bi=0; Bi<B->shape[2]; Bi++) {
                for(int Bj=0; Bj<B->shape[3]; Bj++) {
                    // Interpolate indices
                    int Ai = (Bi * A_hc) / B->shape[2] + coords_from_y;
                    int Aj = (Bj * A_wc) / B->shape[3] + coords_from_x;

                    int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                    int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];

                    B->ptr[B_pos] = A->ptr[A_pos];
                }
            }
        }

    }
}


void cpu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        // Compute random coordinates
        int x1 = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));
        int x2 = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));
        int y1 = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));
        int y2 = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));

        int coords_from_x = std::min(x1, x2);
        int coords_to_x = std::max(x1, x2);
        int coords_from_y = std::min(y1, y2);
        int coords_to_y = std::max(y1, y2);

        for(int c=0; c<B->shape[1]; c++) {
            for(int Bi=0; Bi<=B->shape[2]; Bi++) {
                for(int Bj=0; Bj<=B->shape[3]; Bj++) {

                    int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                    if (Bi >= coords_from_y && Bi <= coords_to_y && Bj >= coords_from_x && Bj <= coords_to_x){
                        B->ptr[B_pos] = constant;
                    } else {
                        B->ptr[B_pos] = A->ptr[B_pos];
                    }

                }
            }
        }

    }
}
