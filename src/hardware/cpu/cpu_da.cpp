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
#include "random.h"


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
    offsets[0] = A->shape[0]/2.0f - B->shape[2]/2.0f+1;
    offsets[1] = A->shape[1]/2.0f - B->shape[3]/2.0f+1;

    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Ai=coords_from[0], Bi=0; Ai<=coords_to[0]; Ai++, Bi++) {
                for(int Aj=coords_from[1], Bj=0; Aj<=coords_to[1]; Aj++, Bj++) {

                    int B_pos = b*B->stride[0] + c*B->stride[1] + (Bi-offsets[0])*B->stride[2] + (Bj-offsets[1])*B->stride[3];
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

    #pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            int A_wc = coords_to[0]-coords_from[0]+1;
            int A_hc = coords_to[0]-coords_from[1]+1;

            for(int Bi=0; Bi<B->shape[2]; Bi++) {
                for(int Bj=0; Bj<B->shape[3]; Bj++) {
                    // Interpolate indices
                    int Ai = (Bi * A_wc) / B->shape[2] + coords_from[0];
                    int Aj = (Bi * A_hc) / B->shape[3] + coords_from[1];

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

    //#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {
                    // Compute random shift
                    int shift_x = (int)(A->shape[2] * uniform(factor_x[0], factor_x[1]));
                    int shift_y = (int)(A->shape[3] * uniform(factor_y[0], factor_y[1]));

                    int Ai = Bi - shift_x;
                    int Aj = Bj - shift_y;

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

    //#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {
                    // Compute random shift
                    // TODO: Center image
                    float scale = uniform(factor[0], factor[1]);
                    int new_shape_x = (int)(A->shape[2] * scale);
                    int new_shape_y = (int)(A->shape[3] * scale);

                    // Interpolate indices
                    int Ai = (Bi * A->shape[2]) / new_shape_x;
                    int Aj = (Bj * A->shape[3]) / new_shape_y;

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

void cpu_flip_random(Tensor *A, Tensor *B, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    //#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            for(int Bi=0; Bi<B->shape[2];Bi++) {
                for(int Bj=0; Bj<B->shape[3];Bj++) {
                    // Apply randomly
                    if (uniform(0.0f, 1.0f) >= 0.5f){
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

}

void cpu_crop_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    // Performs a crop with padding
    int offsets[2] = {0, 0};
    offsets[0] = A->shape[0]/2.0f - B->shape[2]/2.0f+1;
    offsets[1] = A->shape[1]/2.0f - B->shape[3]/2.0f+1;

    //#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            // Compute random coordinates
            int x1 = (int)(A->shape[2] * uniform(factor_x[0], factor_x[1]));
            int x2 = (int)(A->shape[2] * uniform(factor_x[0], factor_x[1]));
            int y1 = (int)(A->shape[3] * uniform(factor_y[0], factor_y[1]));
            int y2 = (int)(A->shape[3] * uniform(factor_y[0], factor_y[1]));

            int coords_from_x = std::min(x1, x2);
            int coords_to_x = std::max(x1, x2);
            int coords_from_y = std::min(y1, y2);
            int coords_to_y = std::max(y1, y2);

            for(int Ai=coords_from_x, Bi=0; Ai<=coords_to_x; Ai++, Bi++) {
                for(int Aj=coords_from_y, Bj=0; Aj<=coords_to_y; Aj++, Bj++) {

                    int B_pos = b*B->stride[0] + c*B->stride[1] + (Bi-offsets[0])*B->stride[2] + (Bj-offsets[1])*B->stride[3];
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

void cpu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    //#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            // Compute random coordinates
            int x1 = (int)(A->shape[2] * uniform(factor_x[0], factor_x[1]));
            int x2 = (int)(A->shape[2] * uniform(factor_x[0], factor_x[1]));
            int y1 = (int)(A->shape[3] * uniform(factor_y[0], factor_y[1]));
            int y2 = (int)(A->shape[3] * uniform(factor_y[0], factor_y[1]));

            int coords_from_x = std::min(x1, x2);
            int coords_to_x = std::max(x1, x2);
            int coords_from_y = std::min(y1, y2);
            int coords_to_y = std::max(y1, y2);

            int A_wc = coords_to_x-coords_from_x+1;
            int A_hc = coords_to_y-coords_from_y+1;

            for(int Bi=0; Bi<B->shape[2]; Bi++) {
                for(int Bj=0; Bj<B->shape[3]; Bj++) {
                    // Interpolate indices
                    int Ai = (Bi * A_wc) / B->shape[2] + coords_from_x;
                    int Aj = (Bi * A_hc) / B->shape[3] + coords_from_y;

                    int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                    int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];

                    B->ptr[B_pos] = A->ptr[A_pos];
                }
            }
        }
    }
}


void cpu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    //#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        for(int c=0; c<B->shape[1]; c++) {

            // Compute random coordinates
            int x1 = (int)(A->shape[2] * uniform(factor_x[0], factor_x[1]));
            int x2 = (int)(A->shape[2] * uniform(factor_x[0], factor_x[1]));
            int y1 = (int)(A->shape[3] * uniform(factor_y[0], factor_y[1]));
            int y2 = (int)(A->shape[3] * uniform(factor_y[0], factor_y[1]));

            int coords_from_x = std::min(x1, x2);
            int coords_to_x = std::max(x1, x2);
            int coords_from_y = std::min(y1, y2);
            int coords_to_y = std::max(y1, y2);

            for(int Bi=coords_from_x; Bi<=coords_to_x; Bi++) {
                for(int Bj=coords_from_y; Bj<=coords_to_y; Bj++) {

                    if (Bi >= 0 && Bi < A->shape[2] && Bj >= 0 && Bj < A->shape[3]){
                        int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                        B->ptr[B_pos] = constant;
                    }

                }
            }
        }
    }

}
