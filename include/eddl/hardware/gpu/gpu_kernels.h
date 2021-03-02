/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#ifndef EDDL_GPU_KERNELS_H
#define EDDL_GPU_KERNELS_H

#include <cuda.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

// Same as in tensor.h
#define GPU_MIN_FLOAT 1.17549e-38f;  // Minimum finite value
#define GPU_MAX_FLOAT 3.40282e+38f;  // Maximum finite value
#define GPU_EPS_FLOAT 1.19209e-07f;  // Machine epsilon (the difference between 1 and the least value greater than 1 that is representable).
#define GPU_LOWEST_FLOAT -3.40282e+38f;  // For floating-point types: implementation-dependent; generally, the negative of max()


// GPU: Core (static)
//void gpu_transpose(Tensor *A, Tensor *B);
//void gpu_copy(Tensor *A, Tensor *B);

__global__ void fill_(float *A, float v, long int size);
__global__ void fill(float *aptr, float *bptr, int t, int aini, int at, int bini, int bt, int tot, int inc);

__global__ void select(float *A, float* B, long int size, int* indices);
__global__ void select_back(float *A, float* B, long int size, int* indices);

__global__ void set_select(float *A, float* B, long int size, int* indices);
__global__ void set_select_back(float *A, float* B, long int size, int* indices);

__global__ void select_rows(float *A, float* B, int rowsize, int size, int* indices, int ini,bool mask_zeros);  // TODO: Legacy
__global__ void deselect_rows(float *A, float* B, int rowsize, int size, int* indices, int ini,int inc,bool mask_zeros);  // TODO: Legacy

__global__ void concat(float *dest, float *src, unsigned int src_size, unsigned int src_stride, unsigned int dest_stride, bool derivative);

// GPU: Create
__global__ void range(float *A, float start, float step, long int size);
__global__ void eye(float *A, long int rows, long int cols, int offset);

__global__ void gpu_diag(float* A, float* B, long int rows, long int cols, int k);

// GPU: Generator
__global__ void init(unsigned int seed, curandState_t* states);
__global__ void random_uniform(curandState_t* states, float* numbers);
//void gpu_rand_signed_uniform(Tensor *A, float v);
//void gpu_rand_binary(Tensor *A, float v);
//void gpu_rand_normal(Tensor *A, float m, float s);

// GPU: Data transformations (2D Optimized) ********************************************
__global__ void shift(float *A, float* B, int batch, int depth, int irows, int icols, int* shift, int mode, float constant);
__global__ void rotate(float *A, float* B, int batch, int depth, int irows, int icols, float angle_rad, int* center, int mode, float constant);
__global__ void scale(float *A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* new_shape, int mode, float constant);
__global__ void flip(float *A, float* B, int batch, int depth, int irows, int icols, int axis);
__global__ void crop(float *A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, int* offsets, float constant, bool inverse);
__global__ void crop_scale(float *A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, int mode, float constant);

// GPU: Data augmentations (2D Optimized) ********************************************
__global__ void shift_random(float *A, float* B, int batch, int depth, int irows, int icols, float* factor_x, float* factor_y, int mode, float constant, float* rnd);
__global__ void rotate_random(float *A, float* B, int batch, int depth, int irows, int icols, float* factor, int* offset_center, int mode, float constant, float* rnd);
__global__ void scale_random(float *A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor, int mode, float constant, float* rnd);
__global__ void flip_random(float *A, float* B, int batch, int depth, int irows, int icols, int axis, float* rnd);
__global__ void crop_random(float *A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* rnd);
__global__ void crop_scale_random(float *A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor, int mode, float constant, float* rnd);
__global__ void cutout_random(float *A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor_x, float* factor_y, float constant, float* rnd);

// GPU: Math (in-place)
__global__ void gpu_abs(float *A, float *B, long int size);
__global__ void gpu_acos(float *A, float *B, long int size);
__global__ void gpu_add(float *A, float *B, long int size, float v);
__global__ void gpu_asin(float *A, float *B, long int size);
__global__ void gpu_atan(float *A, float *B, long int size);
__global__ void gpu_ceil(float *A, float *B, long int size);
__global__ void gpu_clamp(float *A, float *B, long int size, float min, float max);
__global__ void gpu_cos(float *A, float *B, long int size);
__global__ void gpu_cosh(float *A, float *B, long int size);
__global__ void gpu_exp(float *A, float *B, long int size);
__global__ void gpu_floor(float *A, float *B, long int size);
__global__ void gpu_log(float *A, float *B, long int size);
__global__ void gpu_log2(float *A, float *B, long int size);
__global__ void gpu_log10(float *A, float *B, long int size);
__global__ void gpu_logn(float *A, float *B, long int size, float n);
__global__ void gpu_mod(float *A, float *B, long int size, float v);
__global__ void gpu_inv(float *A, float *B,  long int size, float v);
__global__ void gpu_mult(float *A, float *B, long int size, float v);
__global__ void gpu_normalize(float *A, float *B, long int size, float min_ori, float max_ori, float min, float max);
__global__ void gpu_pow(float *A, float *B, long int size, float exp);
__global__ void gpu_powb(float *A, float *B, long int size, float base);
__global__ void gpu_remainder(float *A, float *B, long int size, float v);
__global__ void gpu_round(float *A, float *B, long int size);
__global__ void gpu_rsqrt(float *A, float *B, long int size);
__global__ void gpu_sigmoid(float *A, float *B, long int size);
__global__ void gpu_sign(float *A, float *B, long int size, float zero_sign);
__global__ void gpu_sin(float *A, float *B, long int size);
__global__ void gpu_sinh(float *A, float *B, long int size);
__global__ void gpu_sqr(float *A, float *B, long int size);
__global__ void gpu_sqrt(float *A, float *B, long int size);
__global__ void gpu_tan(float *A, float *B, long int size);
__global__ void gpu_tanh(float *A, float *B, long int size);
__global__ void gpu_trunc(float *A, float *B, long int size);

// GPU: Math (static)
__global__ void gpu_add(float scA, float *A, float scB, float *B, float *C, long int incC, long int size);
//void gpu_inc(Tensor *A, Tensor *B);
//void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
__global__ void gpu_el_mult(float *A,float *B, float *C, long int incC, long int size);
__global__ void gpu_el_div(float *A, float *B, float *C, long int incC, long int size);
__global__ void gpu_sum2D_rowwise(float *A, float* b, float* c, long int cols, long int rows);  //gpu_sum2D_rowwise
__global__ void gpu_sum2D_colwise(float *A, float* b, float* c, long int cols, long int rows); //gpu_sum2D_rowwise
__global__ void gpu_reduce_sum2D(float *A,float *B,long int r,long int c,long int axis);

__global__ void gpu_maximum(float *A, float *B, float v, long int size);
__global__ void gpu_maximum(float *A, float *B, float *C, long int size);
__global__ void gpu_minimum(float *A, float *B, float v, long int size);
__global__ void gpu_minimum(float *A, float *B, float *C, long int size);

// GPU: Should be reductions

// GPU: Reduction
__global__ void gpu_max_d(float *D, float *PD, float *map, int size, int reduction_size, bool argmax);
__global__ void gpu_max(float *A, float *B, int *map, int size, int size_reduction, bool argmax);
__global__ void gpu_min(float *A, float *B, int *map, int size, int size_reduction, bool argmin);
__global__ void gpu_sum(float *A,float *B,int *map, int size, int size_reduction);
__global__ void gpu_sum_abs(float *A, float *B, int *map, int size, int size_reduction);
__global__ void gpu_prod(float *A, float *B, int *map, int size, int size_reduction);
__global__ void gpu_mean(float *A, float *B, int *map, int size, int size_reduction);
__global__ void gpu_median(float *A, float *B, int *map, int size, int size_reduction, float *aux);
__global__ void gpu_var(float *A, float *B, int *map, int size, int size_reduction, bool unbiased);
__global__ void gpu_mode(float *A, float *B, int *map, int size, int size_reduction);


// Previous reductions
__global__ void reduce_mean(float *A,float *B,int *map,int size);
__global__ void reduce_op_sum(float *A,float *B,int *map,int size);
__global__ void reduce_op_diff(float *A,float *B,int *map,int size);
__global__ void reduce_op_mult(float *A,float *B,int *map,int size);
__global__ void reduce_op_div(float *A,float *B,int *map,int size);

__global__ void reduction_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int rs);
__global__ void reduction_back_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int rs);

__global__ void reduction_permute(float *I,float *O,int *ind,int size);
__global__ void reduction_kernel_keep(float *r, float *I, int *ind, int size, int rsize);
__global__ void reduction_kernel_keep_inc(float *r, float *I, int *ind, int size, int rsize);

__global__ void reduction_kernel_sum(float *I,float *O,int m, int d,int *ind,int rs);

// GPU: Linear algebra
__global__ void gpu_norm_fro(float *A, float *B, int *map, int size, int size_reduction);

// Generating index arrays *****************************
__global__ void gpu_where(float *condition, float *A, float *B, float *C, long int size);

// GPU: Logic functions: Comparisons
__global__ void gpu_isfinite(float *A, float *B, long int size);
__global__ void gpu_isinf(float *A, float *B, long int size);
__global__ void gpu_isnan(float *A, float *B, long int size);
__global__ void gpu_isneginf(float *A, float *B, long int size);
__global__ void gpu_isposinf(float *A, float *B, long int size);

// GPU: Logic functions: Comparisons
__global__ void gpu_logical_and(float *A, float *B, float *C, long int size);
__global__ void gpu_logical_or(float *A, float *B, float *C, long int size);
__global__ void gpu_logical_not(float *A, float *B, long int size);
__global__ void gpu_logical_xor(float *A, float *B, float *C, long int size);

// GPU: Logic operations: Comparison ops
__global__ void gpu_allclose(float *A, float *B, float rtol, float atol, bool equal_nan, long int size, bool &close); // TODO: review return
__global__ void gpu_isclose(float *A, float *B, float *C, float rtol, float atol, bool equal_nan, long int size);

__global__ void gpu_greater(float *A, float *B, float v, long int size);
__global__ void gpu_greater(float *A, float *B, float *C, long int size);
__global__ void gpu_greater_equal(float *A, float *B, float v, long int size);
__global__ void gpu_greater_equal(float *A, float *B, float *C, long int size);
__global__ void gpu_less(float *A, float *B, float v, long int size);
__global__ void gpu_less(float *A, float *B, float *C, long int size);
__global__ void gpu_less_equal(float *A, float *B, float v, long int size);
__global__ void gpu_less_equal(float *A, float *B, float *C, long int size);
__global__ void gpu_equal(float *A, float *B, float v, long int size);
__global__ void gpu_equal(float *A, float *B, float *C, long int size);
__global__ void gpu_not_equal(float *A, float *B, float v, long int size);
__global__ void gpu_not_equal(float *A, float *B, float *C, long int size);


// Legacy

__global__ void mask(float *A, float v, long int size);

#endif
