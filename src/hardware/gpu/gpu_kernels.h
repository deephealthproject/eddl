/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#ifndef _TENSOR_CUDA_KERNELS_
#define _TENSOR_CUDA_KERNELS_

#include <cuda.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

// GPU: Comparison

// GPU: Create
__global__ void range(float* a, float start, float step, long int size);
__global__ void eye(float* a, long int rows, long int cols);

// GPU: Core (static)
__global__ void fill(float *aptr,float *bptr,int t,int aini,int at,int bini,int bt,int tot,int inc);
__global__ void fill_(float* a, float v, long int size);
__global__ void mask(float* a, float v, long int size);

// GPU: Transformations
__global__ void shift(float* A, float* B, int batch, int depth, int irows, int icols, int* shift, int mode, float constant);
__global__ void rotate(float* A, float* B, int batch, int depth, int irows, int icols, float angle_rad, int* center, int mode, float constant);
__global__ void scale(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* new_shape, int mode, float constant);
__global__ void flip(float* A, float* B, int batch, int depth, int irows, int icols, int axis, bool apply);
__global__ void crop(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, int* offsets, float constant, bool inverse);
__global__ void crop_scale(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, int mode, float constant);

// GPU: Data augmentation
__global__ void shift_random(float* A, float* B, int batch, int depth, int irows, int icols, float* factor_x, float* factor_y, int mode, float constant, float* rnd);
__global__ void rotate_random(float* A, float* B, int batch, int depth, int irows, int icols, float* factor, int* offset_center, int mode, float constant, float* rnd);
__global__ void scale_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor, int mode, float constant, float* rnd);
__global__ void flip_random(float* A, float* B, int batch, int depth, int irows, int icols, int axis, float* rnd);
__global__ void crop_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* rnd);
__global__ void crop_scale_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor, int mode, float constant, float* rnd);
__global__ void cutout_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor_x, float* factor_y, float constant, float* rnd);

// GPU: Generator
__global__ void init(unsigned int seed, curandState_t* states);
__global__ void random_uniform(curandState_t* states, float* numbers);

// GPU: Math (in-place)
__global__ void abs_(float* a, long int size);
__global__ void acos_(float* a, long int size);
__global__ void add_(float* a, long int size, float v);
__global__ void asin_(float* a, long int size);
__global__ void atan_(float* a, long int size);
__global__ void ceil_(float* a, long int size);
__global__ void clamp_(float* a, long int size, float min, float max);
__global__ void cos_(float* a, long int size);
__global__ void cosh_(float* a, long int size);
__global__ void exp_(float* a, long int size);
__global__ void floor_(float* a, long int size);
__global__ void log_(float* a, long int size);
__global__ void log2_(float* a, long int size);
__global__ void log10_(float* a, long int size);
__global__ void logn_(float* a, long int size, float n);
__global__ void mod_(float* a, long int size, float v);
__global__ void inv_(float* a, long int size);
__global__ void mult_(float* a, long int size, float v);
__global__ void normalize_(float* a, long int size, float min_ori, float max_ori, float min, float max);
__global__ void pow_(float* a, long int size, float exp);
__global__ void powb_(float* a, long int size, float base);
__global__ void reciprocal_(float* a, long int size);
__global__ void remainder_(float* a, long int size, float v);
__global__ void round_(float* a, long int size);
__global__ void rsqrt_(float* a, long int size);
__global__ void sigmoid_(float* a, long int size);
__global__ void sign_(float* a, long int size);
__global__ void sin_(float* a, long int size);
__global__ void sinh_(float* a, long int size);
__global__ void sqr_(float* a, long int size);
__global__ void sqrt_(float* a, long int size);
__global__ void tan_(float* a, long int size);
__global__ void tanh_(float* a, long int size);
__global__ void trunc_(float* a, long int size);

// GPU: Math (static)
__global__ void addc(float scA,float* a,float scB,float *b, float *c,long int incC, long int size);
__global__ void el_mult(float* a,float *b, float *c, long int incC, long int size);
__global__ void el_div(float* a, float *b, float *c, long int incC, long int size);
__global__ void sum_mat_row(float* a, float* b, float* c, long int cols, long int rows);
__global__ void sum_mat_col(float* a, float* b, float* c, long int cols, long int rows);

// GPU: Should be reductions

// GPU: Reduction
__global__ void reduce_sum2D(float *a,float *b,long int r,long int c,long int axis);
__global__ void reduceToSum(float *a, float *b, long int b_size, int a_axis_ndim);

__global__ void reduction_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int rs);
__global__ void reduction_back_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int rs);

__global__ void reduction_permute(float *I,float *O,int *ind,int size);
__global__ void reduction_kernel_keep(float *r, float *I, int *ind, int size, int rsize);
__global__ void reduction_kernel_keep_inc(float *r, float *I, int *ind, int size, int rsize);

__global__ void reduction_kernel_sum(float *I,float *O,int m, int d,int *ind,int rs);







#endif
