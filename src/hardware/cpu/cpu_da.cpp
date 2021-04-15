/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#include <iostream>
#include <utility>
#include <cmath>

#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/random.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void cpu_single_shift(int b, Tensor *A, Tensor *B, vector<int> shift, int wrapping_mode, float constant){
    _profile(_CPU_SINGLE_SHIFT, 0);
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
                    if(wrapping_mode == WrappingMode::Constant){  // Constant
                        B->ptr[B_pos] = constant;
                    }else if(wrapping_mode == WrappingMode::Original){  // Original
                        B->ptr[B_pos] = A->ptr[B_pos];
                    }else{
                        msg("wrapping_mode (" + to_string(wrapping_mode) + ") not implemented", "Tensor::cpu_single_shift");
                    }
                }

            }
        }
    }
    _profile(_CPU_SINGLE_SHIFT, 1);
}

void cpu_single_rotate(int b, Tensor *A, Tensor *B, float angle, vector<int> offset_center, int wrapping_mode, float constant){
    _profile(_CPU_SINGLE_ROTATE, 0);
    float side_a = A->shape[2]/2.0f;
    float side_b = A->shape[3]/2.0f;
    int center[2] = {(int)side_a+offset_center[0], (int)side_b+offset_center[1]};
    float angle_rad = (float)((-angle) * M_PI/180.0f);  // Convert to radians


    for(int c=0; c<B->shape[1]; c++) {
        for (int Bi = 0; Bi < B->shape[2]; Bi++) {
            for (int Bj = 0; Bj < B->shape[3]; Bj++) {
                int Bi_c = Bi - center[0];
                int Bj_c = Bj - center[1];

                int Ai = ::sinf(angle_rad) * Bj_c + ::cosf(angle_rad) * Bi_c + center[0];
                int Aj = ::cosf(angle_rad) * Bj_c - ::sinf(angle_rad) * Bi_c + center[1];

                int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];
                if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]){
                    int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                    B->ptr[B_pos] = A->ptr[A_pos];;
                }else{
                    if(wrapping_mode == WrappingMode::Constant){  // Constant
                        B->ptr[B_pos] = constant;
                    }else if(wrapping_mode == WrappingMode::Original){  // Original
                        B->ptr[B_pos] = A->ptr[B_pos];
                    }else{
                        msg("wrapping_mode (" + to_string(wrapping_mode) + ") not implemented", "Tensor::cpu_single_rotate");
                    }
                }
            }
        }
    }
    _profile(_CPU_SINGLE_ROTATE, 1);
}

void cpu_single_scale(int b, int* offsets, Tensor *A, Tensor *B, vector<int> new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    _profile(_CPU_SINGLE_SCALE, 0);

    for(int c=0; c<B->shape[1]; c++) {

        for(int Bi=0; Bi<B->shape[2];Bi++) {
            for(int Bj=0; Bj<B->shape[3];Bj++) {
                    int Ai = (Bi + offsets[0]);
                    int Aj = (Bj + offsets[1]);

                    if (coordinate_transformation_mode==TransformationMode::HalfPixel) {
                        float scale_y = (float) new_shape[0] / A->shape[2];
                        float scale_x = (float) new_shape[1] / A->shape[3];
                        Ai = ((float)Ai + 0.5f) / scale_y - 0.5f;
                        Aj = ((float)Aj + 0.5f) / scale_x - 0.5f;
                    } else if (coordinate_transformation_mode==TransformationMode::Asymmetric) {
                        float scale_y = (float) new_shape[0] / A->shape[2];
                        float scale_x = (float) new_shape[1] / A->shape[3];
                        Ai = Ai / scale_y;
                        Aj = Aj / scale_x;
                    } else if (coordinate_transformation_mode==TransformationMode::AlignCorners) {
                        Ai = Ai * (A->shape[2]-1) / (new_shape[0] - 1);
                        Aj = Aj * (A->shape[3]-1) / (new_shape[3] - 1);
                    }else {
                        msg("coordinate_transformation_mode (" + to_string(coordinate_transformation_mode) + ") not implemented", "Tensor::cpu_single_scale");
                    }

                    int B_pos = b * B->stride[0] + c * B->stride[1] + Bi * B->stride[2] + Bj * B->stride[3];
                    if (Ai >= 0 && Ai < A->shape[2] && Aj >= 0 && Aj < A->shape[3]) {
                        int A_pos = b * A->stride[0] + c * A->stride[1] + Ai * A->stride[2] + Aj * A->stride[3];
                        B->ptr[B_pos] = A->ptr[A_pos];
                    } else {

                        if(wrapping_mode == WrappingMode::Constant){  // Constant
                            B->ptr[B_pos] = constant;
                        }else if(wrapping_mode == WrappingMode::Original){  // Original
                            B->ptr[B_pos] = A->ptr[B_pos];
                        }else{
                            msg("wrapping_mode (" + to_string(wrapping_mode) + ") not implemented", "Tensor::cpu_single_scale");
                        }

                    }
            }
        }
    }
    _profile(_CPU_SINGLE_SCALE, 1);
}

void cpu_single_flip(int b, bool apply, Tensor *A, Tensor *B, int axis){
    _profile(_CPU_SINGLE_FLIP, 0);

    for(int c=0; c<B->shape[1]; c++) {
        for(int Bi=0; Bi<B->shape[2];Bi++) {
            for(int Bj=0; Bj<B->shape[3];Bj++) {
                int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];

                if(apply){
                    int pos[2] = {Bi, Bj}; pos[axis] = (B->shape[axis+2]-1) - pos[axis];
                    int Ai = pos[0]; int Aj = pos[1];
                    int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                    B->ptr[B_pos] = A->ptr[A_pos];
                }else{
                    B->ptr[B_pos] = A->ptr[B_pos];
                }

            }
        }
    }
    _profile(_CPU_SINGLE_FLIP, 1);
}

void cpu_single_crop(int b, const int* offsets, Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
    _profile(_CPU_SINGLE_CROP, 0);
    for(int c=0; c<B->shape[1]; c++) {
        for(int Bi=0; Bi<B->shape[2];Bi++) {
            for(int Bj=0; Bj<B->shape[3];Bj++) {

                // Compute coordinates
                int Ai = Bi + offsets[0];  // Start from the (0,0) of the cropping area
                int Aj = Bj + offsets[1];

                bool inRegion = Ai >= coords_from[0] && Ai <= coords_to[0] && Aj >= coords_from[1] && Aj <= coords_to[1];
                int B_pos = b*B->stride[0] + c*B->stride[1] + Bi*B->stride[2] + Bj*B->stride[3];  // We always walk through the whole B tensor

                if ((inRegion && !inverse) || (!inRegion && inverse)){
                    int A_pos = b*A->stride[0] + c*A->stride[1] + Ai*A->stride[2] + Aj*A->stride[3];
                    B->ptr[B_pos] = A->ptr[A_pos];
                }else{
                    B->ptr[B_pos] = constant;
                }
            }
        }
    }
    _profile(_CPU_SINGLE_CROP, 1);
}

void cpu_single_crop_scale(int b, Tensor* A, Tensor* B, vector<int> coords_from, vector<int> coords_to, int wrapping_mode, float constant){
    _profile(_CPU_SINGLE_CROP_SCALE, 0);
    int A_hc = coords_to[0]-coords_from[0]+1;
    int A_wc = coords_to[1]-coords_from[1]+1;

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
    _profile(_CPU_SINGLE_CROP_SCALE, 1);
}

// CPU: Data augmentation (2D Optimized) ********************************************
void cpu_shift(Tensor *A, Tensor *B, vector<int> shift, int wrapping_mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
    _profile(_CPU_SHIFT, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        cpu_single_shift(b, A, B, shift, wrapping_mode, constant);
    }
    _profile(_CPU_SHIFT, 1);
}

void cpu_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int wrapping_mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    _profile(_CPU_ROTATE, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        cpu_single_rotate(b, A, B, angle, offset_center, wrapping_mode, constant);
    }
    _profile(_CPU_ROTATE, 1);
}

void cpu_scale(Tensor *A, Tensor *B, vector<int> new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    // I use "new_shape" because I might want to keep the shape of B, but thinking of it as a bigger/smaller matrix
    // If the new_shape is smaller than B, performs a downscale with padding
    // For cases:
    // A=5x5; B=10x10; new_size=10x10 => Normal zoom
    // A=5x5; B=5x5; new_size=5x5 => Normal zoom-out
    // A=10x10; B=10x10; new_size=5x5 => Zoom-out centered
    // A=5x5; B=5x5; new_size=10x10 => Zoom in window

    _profile(_CPU_SCALE, 0);
    // Center crop (if the the crop is smaller than B)
    int offsets[2] = {0, 0};
    offsets[0] = (new_shape[0] - B->shape[2])/2.0f;
    offsets[1] = (new_shape[1] - B->shape[3])/2.0f;

#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        cpu_single_scale(b, offsets, A, B, new_shape, wrapping_mode, constant, coordinate_transformation_mode);
    }
    _profile(_CPU_SCALE, 1);
}

void cpu_flip(Tensor *A, Tensor *B, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    _profile(_CPU_FLIP, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        cpu_single_flip(b, true, A, B, axis);
    }
    _profile(_CPU_FLIP, 1);
}

void cpu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
    // Two cases:
    // -> A=10x10; B=3x3; crop_size=3x3 => Normal crop
    // -> A=10x10; B=10x10; crop_size=3x3 => Crop with padding (inverse of cutout)
    // Inverse => For cutout

    _profile(_CPU_CROP, 0);
    int offsets[2] = {0, 0};
    if(!Tensor::sameShape(A, B)){
        offsets[0] = coords_from[0];
        offsets[1] = coords_from[1];
    }

#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        cpu_single_crop(b, offsets, A, B, coords_from, coords_to, constant, inverse);
    }
    _profile(_CPU_CROP, 1);
}


void cpu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int wrapping_mode, float constant){
    _profile(_CPU_CROP_SCALE, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        cpu_single_crop_scale(b, A, B, coords_from, coords_to, wrapping_mode, constant);
    }
    _profile(_CPU_CROP_SCALE, 1);
}

void cpu_pad(Tensor *A, Tensor *B, vector<int> pads){
//    _profile(_CPU_PAD, 0);
    #pragma omp parallel for
    for(int b=0; b<A->shape[0]; b++) {
        for(int c=0; c<A->shape[1]; c++) {
            for(int h=0; h<A->shape[2]; h++) {
                for(int w=0; w<A->shape[3]; w++) {
                    int A_pos = b*A->stride[0] + c*A->stride[1] + h*A->stride[2] + w*A->stride[3];
                    int B_pos = b*B->stride[0] + c*B->stride[1] + (h+pads[0])*B->stride[2] + (w+pads[3])*B->stride[3];
                    B->ptr[B_pos] = A->ptr[A_pos];
                }
            }
        }
    }
//    _profile(_CPU_PAD, 1);
}

void cpu_pad_back(Tensor *A, Tensor *B, vector<int> pads){
//    _profile(_CPU_PAD, 0);
//    #pragma omp parallel for
    for(int b=0; b<A->shape[0]; b++) {
        for(int c=0; c<A->shape[1]; c++) {
            for(int h=0; h<A->shape[2]; h++) {
                for(int w=0; w<A->shape[3]; w++) {
                    int A_pos = b*A->stride[0] + c*A->stride[1] + h*A->stride[2] + w*A->stride[3];
                    int B_pos = b*B->stride[0] + c*B->stride[1] + (h+pads[0])*B->stride[2] + (w+pads[3])*B->stride[3];
                    A->ptr[A_pos] += B->ptr[B_pos];
                }
            }
        }
    }
//    _profile(_CPU_PAD, 1);
}

// CPU: Data augmentation (2D Optimized) ********************************************
void cpu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int wrapping_mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html

    _profile(_CPU_SHIFT_RANDOM, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        int shift_y = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));
        int shift_x = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));

        cpu_single_shift(b, A, B, {shift_y, shift_x}, wrapping_mode, constant);
    }
    _profile(_CPU_SHIFT_RANDOM, 1);
}

void cpu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int wrapping_mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    _profile(_CPU_ROTATE_RANDOM, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        float angle =  uniform(factor[0], factor[1]);
        cpu_single_rotate(b, A, B, angle, offset_center, wrapping_mode, constant);
    }
    _profile(_CPU_ROTATE_RANDOM, 1);
}

void cpu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int wrapping_mode, float constant, int coordinate_transformation_mode){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    // I use "new_shape" because I might want to keep the shape of B, but thinking of it as a bigger/smaller matrix
    // If the factor is less than 1.0f, performs a downscale with padding

    _profile(_CPU_SCALE_RANDOM, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        float scale = uniform(factor[0], factor[1]);
        int new_shape_y = (int)(A->shape[2] * scale);
        int new_shape_x = (int)(A->shape[3] * scale);

        // Center crop (if the if the crop is smaller than B)
        int offsets[2] = {0, 0};
        offsets[0] = (new_shape_y - A->shape[2])/2.0f;
        offsets[1] = (new_shape_x - A->shape[3])/2.0f;

        cpu_single_scale(b, offsets, A, B, {new_shape_y, new_shape_x}, wrapping_mode, constant, coordinate_transformation_mode);
    }
    _profile(_CPU_SCALE_RANDOM, 1);
}

void cpu_flip_random(Tensor *A, Tensor *B, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html


    _profile(_CPU_FLIP_RANDOM, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {
        bool apply = uniform(0.0f, 1.0f) >= 0.5f;
        cpu_single_flip(b, apply, A, B, axis);
    }
    _profile(_CPU_FLIP_RANDOM, 1);
}


void cpu_crop_random(Tensor *A, Tensor *B){
    // Performs a crop with padding (Keeps the original size)

    _profile(_CPU_CROP_RANDOM, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        // Compute random coordinates
        int w = B->shape[3];
        int h = B->shape[2];
        int x = (int)((A->shape[3]-w) * uniform(0.0f, 1.0f));
        int y = (int)((A->shape[2]-h) * uniform(0.0f, 1.0f));

        int coords_from_x = x;
        int coords_to_x = x+w;
        int coords_from_y = y;
        int coords_to_y = y+h;

        int offsets[2] = {0, 0};
        offsets[0] = coords_from_y;
        offsets[1] = coords_from_x;

        cpu_single_crop(b, offsets, A, B, {coords_from_y, coords_from_x}, {coords_to_y, coords_to_x}, 0.0f, false);
    }
    _profile(_CPU_CROP_RANDOM, 1);
}

void cpu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int wrapping_mode, float constant){

    _profile(_CPU_CROP_SCALE_RANDOM, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        // Compute random coordinates
        float scale = uniform(factor[0], factor[1]);
        int h = (int)(A->shape[2] * scale);
        int w = (int)(A->shape[3] * scale);
        int y = (int)((A->shape[2]-h) * uniform(0.0f, 1.0f));
        int x = (int)((A->shape[3]-w) * uniform(0.0f, 1.0f));

        int coords_from_x = x;
        int coords_to_x = x+w;
        int coords_from_y = y;
        int coords_to_y = y+h;

        cpu_single_crop_scale(b, A, B, {coords_from_y, coords_from_x}, {coords_to_y, coords_to_x}, wrapping_mode, constant);
    }
    _profile(_CPU_CROP_SCALE_RANDOM, 1);
}


void cpu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    // Performs a crop with padding (Keeps the original size)

    _profile(_CPU_CUTOUT_RANDOM, 0);
#pragma omp parallel for
    for(int b=0; b<B->shape[0]; b++) {

        // Compute random coordinates
        int h = (int)(A->shape[2] * uniform(factor_y[0], factor_y[1]));
        int w = (int)(A->shape[3] * uniform(factor_x[0], factor_x[1]));
        int y = (int)((A->shape[2]-h) * uniform(0.0f, 1.0f));
        int x = (int)((A->shape[3]-w) * uniform(0.0f, 1.0f));

        int coords_from_x = x;
        int coords_to_x = x+w;
        int coords_from_y = y;
        int coords_to_y = y+h;

        int offsets[2] = {0, 0};
        cpu_single_crop(b, offsets, A, B, {coords_from_y, coords_from_x}, {coords_to_y, coords_to_x}, constant, true);
    }
    _profile(_CPU_CUTOUT_RANDOM, 0);
}
