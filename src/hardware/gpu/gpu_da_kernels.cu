/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#define _USE_MATH_DEFINES
#include <cmath>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include "eddl/hardware/gpu/gpu_kernels.h"

__device__ void gpu_single_shift(long int thread_id_x, float* A, float* B, int batch, int depth, int irows, int icols, int* shift, int wrapping_mode, float constant){
    int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
    int *B_stride = A_stride;

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % depth;
    int Bi = thread_id_x / B_stride[2] % irows;
    int Bj = thread_id_x / B_stride[3] % icols;
    //--------------
    //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);

    int Ai = Bi - shift[0];
    int Aj = Bj - shift[1];

    if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols){
        int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
        B[thread_id_x] = A[A_pos];
    }else{
        if(wrapping_mode==0){ // Constant
            B[thread_id_x] = constant;
        }else if(wrapping_mode == 5){  // Original
            B[thread_id_x] = A[thread_id_x];
        }else{
            printf("wrapping_mode (%d) not implemented (%s)", wrapping_mode, "Tensor::gpu_single_shift");
        }
    }
}


__device__ void gpu_single_rotate(long int thread_id_x, float* A, float* B, int batch, int depth, int irows, int icols, float angle_rad, int* center, int wrapping_mode, float constant){
    int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
    int *B_stride = A_stride;

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % depth;
    int Bi = thread_id_x / B_stride[2] % irows;
    int Bj = thread_id_x / B_stride[3] % icols;
    //--------------
    //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);

    int Bi_c = Bi - center[0];
    int Bj_c = Bj - center[1];
    int Ai = sinf(angle_rad) * Bj_c + cosf(angle_rad) * Bi_c + center[0];
    int Aj = cosf(angle_rad) * Bj_c - sinf(angle_rad) * Bi_c + center[1];

    if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols){
        int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
        B[thread_id_x] = A[A_pos];
    }else{
        if(wrapping_mode==0){ // Constant
            B[thread_id_x] = constant;
        }else if(wrapping_mode == 5){  // Original
            B[thread_id_x] = A[thread_id_x];
        }else{
            printf("wrapping_mode (%d) not implemented (%s)\n", wrapping_mode, "Tensor::gpu_single_rotate");
        }
    }
}


__device__ void gpu_single_scale(long int thread_id_x, float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    int offsets[2] = {0, 0};
    offsets[0] = (new_shape[0] - orows)/2.0f;
    offsets[1] = (new_shape[1] - ocols)/2.0f;

    int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
    int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % depth;
    int Bi = thread_id_x / B_stride[2] % orows;
    int Bj = thread_id_x / B_stride[3] % ocols;
    //--------------
    //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);


    int Ai = (Bi + offsets[0]);
    int Aj = (Bj + offsets[1]);

    // Select transformation mode: HalfPixel=0, PytorchHalfPixel=1, AlignCorners=2, Asymmetric=3, TFCropAndResize=4
    if (coordinate_transformation_mode==0) {
        float scale_y = (float) new_shape[0] / irows;
        float scale_x = (float) new_shape[1] / icols;
        Ai = ((float)Ai + 0.5f) / scale_y - 0.5f;
        Aj = ((float)Aj + 0.5f) / scale_x - 0.5f;
    } else if (coordinate_transformation_mode==2) {
        float scale_y = (float)(new_shape[0]-1) / (irows - 1);
        float scale_x = (float)(new_shape[1]-1) / (icols - 1);
        Ai = Ai / scale_y;
        Aj = Aj / scale_x;
    } else if (coordinate_transformation_mode==3) {
        float scale_y = (float) new_shape[0] / irows;
        float scale_x = (float) new_shape[1] / icols;
        Ai = Ai / scale_y;
        Aj = Aj / scale_x;
    }
    // If the mode does not exists, must be catched before calling this function


    if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols) {
        int A_pos = b * A_stride[0] + c * A_stride[1] + Ai * A_stride[2] + Aj * A_stride[3];
        B[thread_id_x] = A[A_pos];
    } else {
        if(wrapping_mode==0){ // Constant
            B[thread_id_x] = constant;
        }else if(wrapping_mode == 5){  // Original
            B[thread_id_x] = A[thread_id_x];
        }else{
            printf("wrapping_mode (%d) not implemented (%s)", wrapping_mode, "Tensor::gpu_single_scale");
        }
    }
}


__device__ void gpu_single_scale_back(long int thread_id_x, float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    int offsets[2] = {0, 0};
    offsets[0] = (new_shape[0] - orows)/2.0f;
    offsets[1] = (new_shape[1] - ocols)/2.0f;

    int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
    int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % depth;
    int Bi = thread_id_x / B_stride[2] % orows;
    int Bj = thread_id_x / B_stride[3] % ocols;
    //--------------
    //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);


    int Ai = (Bi + offsets[0]);
    int Aj = (Bj + offsets[1]);

    // Select transformation mode: HalfPixel=0, PytorchHalfPixel=1, AlignCorners=2, Asymmetric=3, TFCropAndResize=4
    if (coordinate_transformation_mode==0) {
        float scale_y = (float) new_shape[0] / irows;
        float scale_x = (float) new_shape[1] / icols;
        Ai = ((float)Ai + 0.5f) / scale_y - 0.5f;
        Aj = ((float)Aj + 0.5f) / scale_x - 0.5f;
    } else if (coordinate_transformation_mode==2) {
        float scale_y = (float)(new_shape[0]-1) / (irows - 1);
        float scale_x = (float)(new_shape[1]-1) / (icols - 1);
        Ai = Ai / scale_y;
        Aj = Aj / scale_x;
    } else if (coordinate_transformation_mode==3) {
        float scale_y = (float) new_shape[0] / irows;
        float scale_x = (float) new_shape[1] / icols;
        Ai = Ai / scale_y;
        Aj = Aj / scale_x;
    }
    // If the mode does not exists, must be catched before calling this function


    if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols) {
        int A_pos = b * A_stride[0] + c * A_stride[1] + Ai * A_stride[2] + Aj * A_stride[3];
        A[A_pos] += B[thread_id_x];
    } else {
        if(wrapping_mode==0){ // Constant
            // B[thread_id_x] = constant;
        }else if(wrapping_mode == 5){  // Original
            A[thread_id_x] += B[thread_id_x];
        }else{
            printf("wrapping_mode (%d) not implemented (%s)", wrapping_mode, "Tensor::gpu_single_scale_back");
        }
    }
}


__device__ void gpu_single_scale3d(long int thread_id_x, float* A, float* B, int batch, int channels, int idepth, int irows, int icols, int odepth, int orows, int ocols, int* new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    int offsets[3] = {0, 0, 0};
    offsets[0] = (new_shape[0] - odepth)/2.0f;
    offsets[1] = (new_shape[1] - orows)/2.0f;
    offsets[2] = (new_shape[2] - ocols)/2.0f;

    int A_stride[5] = {channels*idepth*irows*icols, idepth*irows*icols, irows*icols, icols, 1};
    int B_stride[5] = {channels*odepth*orows*ocols, odepth*orows*ocols, orows*ocols, ocols, 1};

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % channels;
    int Bk = thread_id_x / B_stride[2] % odepth;
    int Bi = thread_id_x / B_stride[3] % orows;
    int Bj = thread_id_x / B_stride[4] % ocols;
    //--------------
    //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);


    int Ak = (Bk + offsets[0]);
    int Ai = (Bi + offsets[1]);
    int Aj = (Bj + offsets[2]);

    // Select transformation mode: HalfPixel=0, PytorchHalfPixel=1, AlignCorners=2, Asymmetric=3, TFCropAndResize=4
    if (coordinate_transformation_mode==0) {
        float scale_z = (float) new_shape[0] / idepth;
        float scale_y = (float) new_shape[1] / irows;
        float scale_x = (float) new_shape[2] / icols;
        Ak = ((float)Ak + 0.5f) / scale_z - 0.5f;
        Ai = ((float)Ai + 0.5f) / scale_y - 0.5f;
        Aj = ((float)Aj + 0.5f) / scale_x - 0.5f;
    } else if (coordinate_transformation_mode==2) {
        float scale_z = (float)(new_shape[0]-1) / (idepth - 1);
        float scale_y = (float)(new_shape[1]-1) / (irows - 1);
        float scale_x = (float)(new_shape[2]-1) / (icols - 1);
        Ak = Ak / scale_z;
        Ai = Ai / scale_y;
        Aj = Aj / scale_x;
    } else if (coordinate_transformation_mode==3) {
        float scale_z = (float) new_shape[0] / idepth;
        float scale_y = (float) new_shape[1] / irows;
        float scale_x = (float) new_shape[2] / icols;
        Ak = Ak / scale_z;
        Ai = Ai / scale_y;
        Aj = Aj / scale_x;
    }
    // If the mode does not exists, must be catched before calling this function


    if (Ak >= 0 && Ak < idepth && Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols) {
        int A_pos = b * A_stride[0] + c * A_stride[1] + Ak * A_stride[2] + Ai * A_stride[3] + Aj * A_stride[4];
        B[thread_id_x] = A[A_pos];
    } else {
        if(wrapping_mode==0){ // Constant
            B[thread_id_x] = constant;
        }else if(wrapping_mode == 5){  // Original
            B[thread_id_x] = A[thread_id_x];
        }else{
            printf("wrapping_mode (%d) not implemented (%s)", wrapping_mode, "Tensor::gpu_single3d_scale");
        }
    }
}


__device__ void gpu_single_scale3d_back(long int thread_id_x, float* A, float* B, int batch, int channels, int idepth, int irows, int icols, int odepth, int orows, int ocols, int* new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    int offsets[3] = {0, 0, 0};
    offsets[0] = (new_shape[0] - odepth)/2.0f;
    offsets[1] = (new_shape[1] - orows)/2.0f;
    offsets[2] = (new_shape[2] - ocols)/2.0f;

    int A_stride[5] = {channels*idepth*irows*icols, idepth*irows*icols, irows*icols, icols, 1};
    int B_stride[5] = {channels*odepth*orows*ocols, odepth*orows*ocols, orows*ocols, ocols, 1};

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % channels;
    int Bk = thread_id_x / B_stride[2] % odepth;
    int Bi = thread_id_x / B_stride[3] % orows;
    int Bj = thread_id_x / B_stride[4] % ocols;
    //--------------
    //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);


    int Ak = (Bk + offsets[0]);
    int Ai = (Bi + offsets[1]);
    int Aj = (Bj + offsets[2]);

    // Select transformation mode: HalfPixel=0, PytorchHalfPixel=1, AlignCorners=2, Asymmetric=3, TFCropAndResize=4
    if (coordinate_transformation_mode==0) {
        float scale_z = (float) new_shape[0] / idepth;
        float scale_y = (float) new_shape[1] / irows;
        float scale_x = (float) new_shape[2] / icols;
        Ak = ((float)Ak + 0.5f) / scale_z - 0.5f;
        Ai = ((float)Ai + 0.5f) / scale_y - 0.5f;
        Aj = ((float)Aj + 0.5f) / scale_x - 0.5f;
    } else if (coordinate_transformation_mode==2) {
        float scale_z = (float)(new_shape[0]-1) / (idepth - 1);
        float scale_y = (float)(new_shape[1]-1) / (irows - 1);
        float scale_x = (float)(new_shape[2]-1) / (icols - 1);
        Ak = Ak / scale_z;
        Ai = Ai / scale_y;
        Aj = Aj / scale_x;
    } else if (coordinate_transformation_mode==3) {
        float scale_z = (float) new_shape[0] / idepth;
        float scale_y = (float) new_shape[1] / irows;
        float scale_x = (float) new_shape[2] / icols;
        Ak = Ak / scale_z;
        Ai = Ai / scale_y;
        Aj = Aj / scale_x;
    }
    // If the mode does not exists, must be catch before calling this function


    if (Ak >= 0 && Ak < idepth && Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols) {
        int A_pos = b * A_stride[0] + c * A_stride[1] + Ak * A_stride[2] + Ai * A_stride[3] + Aj * A_stride[4];
        A[A_pos] += B[thread_id_x];
    } else {
        if(wrapping_mode==0){ // Constant
            // B[thread_id_x] = constant;
        }else if(wrapping_mode == 5){  // Original
            A[thread_id_x] += B[thread_id_x];
        }else{
            printf("wrapping_mode (%d) not implemented (%s)", wrapping_mode, "Tensor::gpu_single_scale3d_back");
        }
    }
}

__device__ void gpu_single_flip(long int thread_id_x, float* A, float* B, int batch, int depth, int irows, int icols, int axis, bool apply){
    int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
    int *B_stride = A_stride;

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % depth;
    int Bi = thread_id_x / B_stride[2] % irows;
    int Bj = thread_id_x / B_stride[3] % icols;
    //--------------
    //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);

    if(apply){
        int pos[2] = {Bi, Bj}; pos[axis] = (irows-1) - pos[axis];
        int Ai = pos[0]; int Aj = pos[1];
        int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
        B[thread_id_x] = A[A_pos];
    }else{
        B[thread_id_x] = A[thread_id_x];
    }
}


__device__ void gpu_single_crop(long int thread_id_x, float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, int* offsets, float constant, bool inverse){
    int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
    int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % depth;
    int Bi = thread_id_x / B_stride[2] % orows;
    int Bj = thread_id_x / B_stride[3] % ocols;

    // Compute coordinates
    int Ai = Bi + offsets[0];  // Start from the (0,0) of the cropping area
    int Aj = Bj + offsets[1];

    bool inRegion = Ai >= coords_from[0] && Ai <= coords_to[0] && Aj >= coords_from[1] && Aj <= coords_to[1];
    if ((inRegion && !inverse) || (!inRegion && inverse)){
        int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
        B[thread_id_x] = A[A_pos];
    }else{
        B[thread_id_x] = constant;
    }
}


__device__ void gpu_single_crop_scale(long int thread_id_x, float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, int wrapping_mode, float constant){
    int A_hc = coords_to[0]-coords_from[0]+1;
    int A_wc = coords_to[1]-coords_from[1]+1;

    int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
    int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

    //--------------
    int b = thread_id_x / B_stride[0] % batch;
    int c = thread_id_x / B_stride[1] % depth;
    int Bi = thread_id_x / B_stride[2] % orows;
    int Bj = thread_id_x / B_stride[3] % ocols;

    int Ai = (Bi * A_hc) / orows + coords_from[0];
    int Aj = (Bj * A_wc) / ocols + coords_from[1];

    int A_pos = b * A_stride[0] + c * A_stride[1] + Ai * A_stride[2] + Aj * A_stride[3];
    B[thread_id_x] = A[A_pos];
}


__global__ void shift(float* A, float* B, int batch, int depth, int irows, int icols, int* shift, int wrapping_mode, float constant){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        gpu_single_shift(thread_id_x, A, B, batch, depth, irows, icols, shift, wrapping_mode, constant);
    }

}


__global__ void rotate(float* A, float* B, int batch, int depth, int irows, int icols, float angle_rad, int* center, int wrapping_mode, float constant){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    // Not implemented
    if (thread_id_x < ops){
        gpu_single_rotate(thread_id_x, A, B, batch, depth, irows, icols, angle_rad, center, wrapping_mode, constant);
    }
}


__global__ void scale(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*orows*ocols;

    if (thread_id_x < ops){
        gpu_single_scale(thread_id_x, A, B, batch, depth, irows, icols, orows, ocols, new_shape, wrapping_mode, constant, coordinate_transformation_mode);
    }

}

__global__ void scale_back(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*orows*ocols;

    if (thread_id_x < ops){
        gpu_single_scale_back(thread_id_x, A, B, batch, depth, irows, icols, orows, ocols, new_shape, wrapping_mode, constant, coordinate_transformation_mode);
    }
}

__global__ void flip(float* A, float* B, int batch, int depth, int irows, int icols, int axis){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        gpu_single_flip(thread_id_x, A, B, batch, depth, irows, icols, axis, true);
    }
}


__global__ void crop(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, int* offsets, float constant, bool inverse){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        gpu_single_crop(thread_id_x, A, B, batch, depth, irows, icols, orows, ocols, coords_from, coords_to, offsets, constant, inverse);
    }
}


__global__ void crop_scale(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, int wrapping_mode, float constant){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        gpu_single_crop_scale(thread_id_x, A, B, batch, depth, irows, icols, orows, ocols, coords_from, coords_to, wrapping_mode, constant);
    }
}

__global__ void gpu_pad(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int padt, int padl){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch*depth*irows*icols;

    if (thread_id_x < ops){
        int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
        int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

        int b = thread_id_x / A_stride[0] % batch;
        int c = thread_id_x / A_stride[1] % depth;
        int Ai = thread_id_x / A_stride[2] % irows;
        int Aj = thread_id_x / A_stride[3] % icols;

        int B_pos = b*B_stride[0] + c*B_stride[1] + (Ai+padt)*B_stride[2] + (Aj+padl)*B_stride[3];
        B[B_pos] = A[thread_id_x];
    }
}

__global__ void gpu_pad_back(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols,  int padt, int padl){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
        int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

        int b = thread_id_x / A_stride[0] % batch;
        int c = thread_id_x / A_stride[1] % depth;
        int Ai = thread_id_x / A_stride[2] % irows;
        int Aj = thread_id_x / A_stride[3] % icols;

        int B_pos = b*B_stride[0] + c*B_stride[1] + (Ai+padt)*B_stride[2] + (Aj+padl)*B_stride[3];
        A[thread_id_x] += B[B_pos];
    }
}

__global__ void shift_random(float* A, float* B, int batch, int depth, int irows, int icols, float* factor_x, float* factor_y, int wrapping_mode, float constant, float* rnd){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        int b = thread_id_x / (depth*irows*icols) % batch;

        int shift_y = (int)(irows * ((factor_y[1]-factor_y[0]) * rnd[b+1] + factor_y[0]));
        int shift_x = (int)(icols * ((factor_x[1]-factor_x[0]) * rnd[b] + factor_x[0]));
        int shift[2] = {shift_y, shift_x};

        gpu_single_shift(thread_id_x, A, B, batch, depth, irows, icols, shift, wrapping_mode, constant);
    }

}

__global__ void rotate_random(float* A, float* B, int batch, int depth, int irows, int icols, float* factor, int* center, int wrapping_mode, float constant, float* rnd){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        int b = thread_id_x / (depth*irows*icols) % batch;

        float angle = -1.0f * ((factor[1]-factor[0]) * rnd[b] + factor[0]);
        float angle_rad = (float)((-angle) * M_PI/180.0f);  // Convert to radians

        gpu_single_rotate(thread_id_x, A, B, batch, depth, irows, icols, angle_rad, center, wrapping_mode, constant);
    }
}

__global__ void scale_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor, int wrapping_mode, float constant, int coordinate_transformation_mode, float* rnd){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*orows*ocols;

    if (thread_id_x < ops){
        int b = thread_id_x / (depth*orows*ocols) % batch;

        float scale = (factor[1]-factor[0]) * rnd[b] + factor[0];
        int new_shape_y = (int)(irows * scale);
        int new_shape_x = (int)(icols * scale);
        int new_shape[2] = {new_shape_y, new_shape_x};

        gpu_single_scale(thread_id_x, A, B, batch, depth, irows, icols, orows, ocols, new_shape, wrapping_mode, constant, coordinate_transformation_mode);
    }

}


__global__ void flip_random(float* A, float* B, int batch, int depth, int irows, int icols, int axis, float* rnd){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        //--------------
        int b = thread_id_x / (depth*irows*icols) % batch;

        bool apply = rnd[b] >= 0.5f;
        gpu_single_flip(thread_id_x, A, B, batch, depth, irows, icols, axis, apply);
    }
}


__global__ void crop_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* rnd){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        int b = thread_id_x / (depth*orows*ocols) % batch;

        // Compute random coordinates
        int w = ocols;
        int h = orows;
        int x = (int)((icols-w) * rnd[b]);
        int y = (int)((irows-h) * rnd[b+1]);

        int coords_from_x = x;
        int coords_to_x = x+w;
        int coords_from_y = y;
        int coords_to_y = y+h;

        int coords_from[2] = {coords_from_y, coords_from_x};
        int coords_to[2] = {coords_to_y, coords_to_x};

        // Compute offsets
        int offsets[2] = {0, 0}; // Used only during the normal crop
        if(irows!=orows || icols!=ocols){
            offsets[0] = coords_from[0];
            offsets[1] = coords_from[1];
        }

        gpu_single_crop(thread_id_x, A, B, batch, depth, irows, icols, orows, ocols, coords_from, coords_to, offsets, 0.0f, false);
    }
}


__global__ void crop_scale_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor, int wrapping_mode, float constant, float* rnd) {
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        //--------------
        int b = thread_id_x / (depth*orows*ocols) % batch;

        // Compute random coordinates
        float scale = ((factor[1]-factor[0]) * rnd[b] + factor[0]);
        int h = (int)(irows * scale);
        int w = (int)(icols * scale);
        int y = (int)((irows-h) * rnd[b+1]);
        int x = (int)((icols-w) * rnd[b+2]);

        int coords_from_x = x;
        int coords_to_x = x+w;
        int coords_from_y = y;
        int coords_to_y = y+h;

        int coords_from[2] = {coords_from_y, coords_from_x};
        int coords_to[2] = {coords_to_y, coords_to_x};

        gpu_single_crop_scale(thread_id_x, A, B, batch, depth, irows, icols, orows, ocols, coords_from, coords_to, wrapping_mode, constant);
    }
}


__global__ void cutout_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor_x, float* factor_y, float constant, float* rnd){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        int b = thread_id_x / (depth*orows*ocols) % batch;

        // Compute random coordinates
        int h = (int)(irows * ((factor_y[1]-factor_y[0]) * rnd[b] + factor_y[0]));
        int w = (int)(icols * ((factor_x[1]-factor_x[0]) * rnd[b+1] + factor_x[0]));
        int y = (int)((irows-h) * rnd[b+2]);
        int x = (int)((icols-w) * rnd[b+3]);

        int coords_from_x = x;
        int coords_to_x = x+w;
        int coords_from_y = y;
        int coords_to_y = y+h;

        int offsets[2] = {0, 0}; // Used only during the normal crop
        int coords_from[2] = {coords_from_y, coords_from_x};
        int coords_to[2] = {coords_to_y, coords_to_x};

        gpu_single_crop(thread_id_x, A, B, batch, depth, irows, icols, orows, ocols, coords_from, coords_to, offsets, constant, true);
    }
}


__global__ void scale3d(float *A, float* B, int batch, int channels, int idepth, int irows, int icols, int odepth, int orows, int ocols, int* new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch*channels*odepth*orows*ocols;

    if (thread_id_x < ops){
        gpu_single_scale3d(thread_id_x, A, B, batch, channels, idepth, irows, icols, odepth, orows, ocols, new_shape, wrapping_mode, constant, coordinate_transformation_mode);
    }
}

__global__ void scale3d_back(float *A, float* B, int batch, int channels, int idepth, int irows, int icols, int odepth, int orows, int ocols, int* new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch*channels*odepth*orows*ocols;

    if (thread_id_x < ops){
        gpu_single_scale3d_back(thread_id_x, A, B, batch, channels, idepth, irows, icols, odepth, orows, ocols, new_shape, wrapping_mode, constant, coordinate_transformation_mode);
    }
}
