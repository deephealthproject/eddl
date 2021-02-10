/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "eddl/hardware/gpu/gpu_kernels.h"

__global__ void gpu_max_d(float *D, float *PD, float *map, int size, int reduction_size, bool argmax){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    if (thread_id_x<size) {
        int offset = thread_id_x*reduction_size;

        // Choose if we're getting the maximum value or the position
        if(argmax) {
            int argmax_addr = map[thread_id_x];
            PD[offset+argmax_addr] += D[thread_id_x];
        }else{
            PD[offset+thread_id_x] += D[thread_id_x];;
        }
    }
}

__global__ void gpu_max(float *A, float *B, int *map, int size, int size_reduction, bool argmax){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];
        float tmp_max = A[*base_addr];
        int tmp_argmax = 0;

        float val;
        for(int i=1; i<size_reduction; i++){
            val = A[*(base_addr+i)];
            if(val > tmp_max){
                tmp_max = val;
                tmp_argmax = i;
            }
        }

        // Choose if we're getting the maximum value or the position
        if(argmax) {
            B[thread_id_x] = (float)tmp_argmax;
        }else{
            B[thread_id_x] = tmp_max;
        }
    }
}

__global__ void gpu_min(float *A, float *B, int *map, int size, int size_reduction, bool argmin){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];
        float tmp_min = A[*base_addr];
        int tmp_argmin = 0;

        float val;
        for(int i=1; i<size_reduction; i++){
            val = A[*(base_addr+i)];
            if(val < tmp_min){
                tmp_min = val;
                tmp_argmin = i;
            }
        }

        // Choose if we're getting the minimum value or the position
        if(argmin) {
            B[thread_id_x] = (float)tmp_argmin;
        }else{
            B[thread_id_x] = tmp_min;
        }
    }
}

__global__ void gpu_sum(float *A, float *B, int *map, int size, int size_reduction){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];

        float tmp = 0.0f;
        for(int i=0; i<size_reduction; i++){
            tmp += A[*(base_addr+i)];
        }

        B[thread_id_x] = tmp;
    }
}

__global__ void gpu_sum_abs(float *A, float *B, int *map, int size, int size_reduction){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];

        float tmp = 0.0f;
        for(int i=0; i<size_reduction; i++){
            tmp += abs(A[*(base_addr+i)]);
        }

        B[thread_id_x] = tmp;
    }
}

__global__ void gpu_prod(float *A, float *B, int *map, int size, int size_reduction){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];

        float tmp = 1.0f;
        for(int i=0; i<size_reduction; i++){
            tmp *= A[*(base_addr+i)];
        }

        B[thread_id_x] = tmp;
    }
}

__global__ void gpu_mean(float *A, float *B, int *map, int size, int size_reduction){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];

        float tmp = 0.0f;
        for(int i=0; i<size_reduction; i++){
            tmp += A[*(base_addr+i)];
        }

        B[thread_id_x] = tmp/(float)size_reduction;
    }
}

__global__ void gpu_median(float *A, float *B, int *map, int size, int size_reduction, float *aux){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        // Copy values
        long int offset = thread_id_x*size_reduction;
        for(int i=0; i<size_reduction; i++){
            aux[offset+i] = A[map[offset+i]];
        }

        // Sort data
        thrust::sort(thrust::device, aux + offset, aux + offset + size_reduction);

        // Get median
        int midpoint = (int)offset + size_reduction/ 2;
        if(size_reduction % 2==1 && size_reduction>1) {
            B[thread_id_x] = aux[midpoint];
        }else{
            B[thread_id_x] = (aux[midpoint-1]+aux[midpoint])/2.0f;
        }

    }
}

__global__ void gpu_var(float *A, float *B, int *map, int size, int size_reduction, bool unbiased){
    // IMPORTANT TRICK: B ALREADY CONTAINS THE MEAN!!!!!!!
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];

        float tmp;
        float sum = 0.0f;
        for(int i=0; i<size_reduction; i++){
            tmp = A[*(base_addr+i)] - B[thread_id_x];
            sum += tmp*tmp;
        }

        if(unbiased){
            B[thread_id_x] = sum/((float)size_reduction-1.0f);
        } else {
            B[thread_id_x] = sum/(float)size_reduction;
        }
    }
}

__global__ void gpu_norm_fro(float *A, float *B, int *map, int size, int size_reduction){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];

        float tmp = 0.0f;
        float val;
        for(int i=0; i<size_reduction; i++){
            val = A[*(base_addr+i)];
            tmp += val*val;
        }

        B[thread_id_x] = sqrt(tmp);
    }
}

__global__ void gpu_mode(float *A, float *B, int *map, int size, int size_reduction){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        int *base_addr = &map[thread_id_x*size_reduction+0];

        // Copy values
        int *values = new int[size_reduction];  // Dynamic allocation is not the best approach
        for(int i=0; i<size_reduction; i++){
            values[i] = (int)A[*(base_addr+i)];
        }

        // Sort data
        thrust::sort(thrust::seq, values, values + size_reduction);

        // Get most frequent element
        int most_frequent_val;
        int most_frequent_times = 0;

        int val = values[0];
        int frequency = 1;
        for(int i=1; i<size_reduction; i++){
            // Check if the value has change
            if(val==values[i]){
                frequency++;
            }else{
                val = values[i];
                frequency = 1;
            }

            // Check frequency
            if(frequency>most_frequent_times){
                most_frequent_val = val;
                most_frequent_times = frequency;
            }
        }
        // Assign most frequent value
        B[thread_id_x] = (float)most_frequent_val;

        // Delete temp array
        delete[] values;
    }
}


/* PREVIOUS REDUCES ***********************************/

__global__ void reduce_mean(float *A,float *B,int *map,int size)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        atomicAdd(&(B[map[thread_id_x]]),A[thread_id_x]);
    }

}

__global__ void reduce_op_sum(float *A,float *B,int *map,int size)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        A[thread_id_x]+=B[map[thread_id_x]];
    }
}

__global__ void reduce_op_diff(float *A,float *B,int *map,int size)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        A[thread_id_x]-=B[map[thread_id_x]];
    }

}
__global__ void reduce_op_mult(float *A,float *B,int *map,int size)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        A[thread_id_x]*=B[map[thread_id_x]];
    }

}
__global__ void reduce_op_div(float *A,float *B,int *map,int size)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size) {
        A[thread_id_x]/=B[map[thread_id_x]];
    }

}


//dim3 dimGrid(RD->index.size());
//dim3 dimBlock(1);
__global__ void reduction_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int rs)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;


    int j;
    float sum=0;
    float v,val;

    int i;

    int p=rs*blockIdx.x;


    for(j=0;j<rs;j++,p++) {
        v=I[ind[p]];
        if (m==2) {
            if (j==0) {val=v;i=p;}
            else if (v>val) {
                val=v;
                i=p;
            }
        }
        else if (m==3) {
            if (j==0) {val=v;i=p;}
            else if (v<val) {
                val=v;
                i=p;
            }
        }
        else sum+=v;
    }

    p=rs*blockIdx.x;
    // set in Output
    if (m<2) { // mean or sum
        if (m==0) sum/=d;
        if (keepdims) {
            for(j=0;j<rs;j++,p++)
                O[ind[p]]=sum;
        }
        else O[thread_id_x]=sum;
    }
    else { // rs or min
        if (keepdims) {
            for(j=0;j<rs;j++,p++) {
                O[ind[p]]=val;
                S[ind[p]]=i;
            }
        }
        else {
            O[thread_id_x]=val;
            S[thread_id_x]=i;
        }
    }

}


//dim3 dimGrid(RD->index.size());
//dim3 dimBlock(1);
__global__ void reduction_back_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int rs)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    int j;
    float val=0;
    int p;

    // set in Delta
    if (m>=2) {
        int p=S[thread_id_x];
        O[p]+=I[thread_id_x];
    }
    else {
        p=rs*blockIdx.x;
        if(keepdims) {
            for(j=0;j<rs;j++,p++)
                val+=I[ind[p]];
        }
        else val=I[thread_id_x];
        if (m==0) val/=d;

        p=rs*blockIdx.x;
        for(j=0;j<rs;j++,p++)
            O[ind[p]]+=val;
    }
}



////////////////////
// FOR SUM and MEAN
// Faster in Conv
///////////////////

//dim3 dimGrid(red_size);
//dim3 dimBlock(RD->index.size());

__global__ void reduction_permute(float *I,float *O,int *ind,int size)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x<size)
        O[thread_id_x]=I[ind[thread_id_x]];
}

__global__ void reduction_kernel_keep(float *red, float *O, int *ind, int size, int rsize)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    if (thread_id_x<size*rsize) {
        O[ind[thread_id_x]]=red[thread_id_x/rsize];
    }
}

__global__ void reduction_kernel_keep_inc(float *red, float *O, int *ind, int size, int rsize)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    if (thread_id_x<size*rsize) {
        O[ind[thread_id_x]]+=red[thread_id_x/rsize];
    }
}
