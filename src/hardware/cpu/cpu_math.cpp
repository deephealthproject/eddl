/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_tensor.h"
#include <unordered_map>

// CPU: Math (in-place) ********************************************

void cpu_abs(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::fabs(A->ptr[i]);
}

void cpu_acos(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::acosf(A->ptr[i]);
}

void cpu_add(Tensor *A, Tensor *B, float v) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = A->ptr[i] + v;
}


void cpu_asin(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::asinf(A->ptr[i]);
}

void cpu_atan(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::atanf(A->ptr[i]);
}

void cpu_ceil(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::ceilf(A->ptr[i]);
}

void cpu_clamp(Tensor *A, Tensor *B, float min, float max){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        if (A->ptr[i] < min){
            B->ptr[i] = min;
        } else if(A->ptr[i] > max){
            B->ptr[i] = max;
        }else {
            B->ptr[i] = A->ptr[i];
        }
    }
}


void cpu_cos(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::cosf(A->ptr[i]);
}

void cpu_cosh(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::coshf(A->ptr[i]);
}

void cpu_exp(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::expf(A->ptr[i]);
}

void cpu_floor(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::floorf(A->ptr[i]);
}

void cpu_inv(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = v/A->ptr[i];
}

void cpu_log(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::logf(A->ptr[i]);
}

void cpu_log2(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::log2f(A->ptr[i]);
}

void cpu_log10(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::log10f(A->ptr[i]);
}

void cpu_logn(Tensor *A, Tensor *B, float n) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::logf(A->ptr[i])/::logf(n);
}


void cpu_mod(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::fmod(A->ptr[i], v);
}

void cpu_mult(Tensor *A, Tensor *B, float v) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = A->ptr[i] * v;
}

void cpu_normalize(Tensor *A, Tensor *B, float min, float max){
    // Normalize in range: 423 from [23, 562], to range [-1, 1] => 0.4842
    // (max2-min2)/(max1-min1) * (x-min1) + min2
    float max_ori = A->max();
    float min_ori = A->min();

#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) {
        B->ptr[i] = (max-min)/(max_ori-min_ori) * (A->ptr[i]-min_ori) + min;
    }
}

void cpu_pow(Tensor *A, Tensor *B, float exp) {
    // To compute the power, std uses real floating-point number with the formurla: e^(y*log_(x))
    // Quite inefficient (x100 slower) in g++ except for pow_(x, 2) which is inlined as x*x
    // speed: 0.057887s
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::powf(A->ptr[i], exp);
}

void cpu_powb(Tensor *A, Tensor *B, float base) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::powf(base, A->ptr[i]);
}

void cpu_remainder(Tensor *A, Tensor *B, float v) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = fmod((v + fmod(A->ptr[i], v)), v);
}

void cpu_round(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::roundf(A->ptr[i]);
}

void cpu_rsqrt(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = 1.0f/::sqrtf(A->ptr[i]);
}

void cpu_sigmoid(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = 1.0f/(1.0f + ::expf(-A->ptr[i]));
}

void cpu_sign(Tensor *A, Tensor *B, float zero_sign){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) {
        if(A->ptr[i] > 0.0f){
            B->ptr[i] = 1.0f;
        }else if(A->ptr[i] < 0.0f){
            B->ptr[i] = -1.0f;
        }else{
            B->ptr[i] = zero_sign;  // 0.0f recommended
        }
    };
}


void cpu_sin(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::sinf(A->ptr[i]);
}

void cpu_sinh(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::sinhf(A->ptr[i]);
}

void cpu_sqr(Tensor *A, Tensor *B) {
    // pow(x, 2) == x*x  To know more, read comments in pow_'s function
    // speed: 0.000497s
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = A->ptr[i] * A->ptr[i];
}

void cpu_sqrt(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::sqrtf(A->ptr[i]);
}

void cpu_tan(Tensor *A, Tensor *B){
//    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i) {
        float r1 = A->ptr[i];
        B->ptr[i] = ::tanf(A->ptr[i]);
        float r2 = B->ptr[i];
        if(::abs(r2)> 1000){
            int asdasd = 33;
        }
    }
}

void cpu_tanh(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::tanhf(A->ptr[i]);
}

void cpu_trunc(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::truncf(A->ptr[i]);
}



// CPU: Math (static) ***************************

void cpu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
#pragma omp parallel for
    for (int i = 0; i < A->size; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] + scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] + scB * B->ptr[i];
}


void cpu_inc(Tensor *A, Tensor *B) {
    B->tsem->lock();

#pragma omp parallel for
    for (int i = 0; i < A->size; i++){
        B->ptr[i] += A->ptr[i];
    }

    B->tsem->unlock();
}

void cpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
    if (!tB) {
        if (!tA) {
            if (!incC) *(C->ptr2) = *(B->ptr2) * (*(A->ptr2));
            else *(C->ptr2) += *(B->ptr2) * (*(A->ptr2));
        } else {
            if (!incC) *(C->ptr2) = *(B->ptr2) * ((*(A->ptr2)).transpose());
            else *(C->ptr2) += *(B->ptr2) * ((*(A->ptr2)).transpose());
        }
    } else {
        if (!tA) {
            if (!incC) *(C->ptr2) = (*(B->ptr2)).transpose() * (*(A->ptr2));
            else *(C->ptr2) += (*(B->ptr2)).transpose() * (*(A->ptr2));
        } else {
            if (!incC) *(C->ptr2) = (*(B->ptr2)).transpose() * ((*(A->ptr2)).transpose());
            else *(C->ptr2) += (*(B->ptr2)).transpose() * ((*(A->ptr2)).transpose());
        }
    }
}

void cpu_el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
#pragma omp parallel for
    for (int i = 0; i < A->size; i++)
        if (incC) C->ptr[i] += A->ptr[i] / B->ptr[i];
        else C->ptr[i] = A->ptr[i] / B->ptr[i];
}


void cpu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
#pragma omp parallel for
    for (int i = 0; i < A->size; i++)
        if (incC) C->ptr[i] += A->ptr[i] * B->ptr[i];
        else C->ptr[i] = A->ptr[i] * B->ptr[i];
}


void cpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
#pragma omp parallel for
    for (int i = 0; i < A->shape[0]; i++) {
        int p=i*A->shape[1];
        for (int j = 0; j < A->shape[1]; j++, p++)
            C->ptr[p] = A->ptr[p] + B->ptr[j];
    }
}

void cpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {

#pragma omp parallel for
    for (int i = 0; i < A->shape[0]; i++) {
        int p=i*A->shape[1];
        for (int j = 0; j < A->shape[1]; j++, p++)
            C->ptr[p] = A->ptr[p] + B->ptr[i];
    }
}


void cpu_maximum(Tensor* A, Tensor* B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) {
        B->ptr[i] = ::max(A->ptr[i], v);
    }
}

void cpu_maximum(Tensor* A, Tensor* B, Tensor* C){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) {
        C->ptr[i] = ::max(A->ptr[i], B->ptr[i]);
    }
}

void cpu_minimum(Tensor* A, Tensor* B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) {
        B->ptr[i] = ::min(A->ptr[i], v);
    }
}

void cpu_minimum(Tensor* A, Tensor* B, Tensor* C){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) {
        C->ptr[i] = ::min(A->ptr[i], B->ptr[i]);
    }
}


// CPU: Should be reductions ***************************


float cpu_max(Tensor *A) {
    auto t = cpu_max(A->ptr, A->size, nullptr);
    return std::get<0>(t);  // get max
}


void cpu_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        auto t = cpu_max(A->ptr, rd->index[i].size(), rd->index[i].data());
        B->ptr[i] = std::get<0>(t);  // get max
    }
}

float cpu_argmax(Tensor *A) {
    auto t = cpu_max(A->ptr, A->size, nullptr);
    return std::get<1>(t);  // get argmax
}


void cpu_argmax(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        auto t = cpu_max(A->ptr, rd->index[i].size(), rd->index[i].data());
        B->ptr[i] = std::get<1>(t);  // get argmax
    }
}


std::tuple<float, int> cpu_max(float *ptr, int size, int *map) {
    float shared_max = MIN_FLOAT;
    int shared_argmax = 0;

        #pragma omp parallel
    {
        float max = MIN_FLOAT;
        int argmax = 0;

        // TODO: I don't like this approach
        if(map == nullptr){
            #pragma omp for nowait
            for (int i = 0; i < size; ++i) {
                if(ptr[i]>max){
                    max = ptr[i];
                    argmax = i;
                }
            }
        }else{
            #pragma omp for nowait
            for (int i = 0; i < size; ++i) {
                if(ptr[i]>max){
                    max = ptr[map[i]];
                    argmax = i;
                }
            }
        }

    #pragma omp critical
        {
            if(max>shared_argmax){
                shared_argmax = max;
                shared_argmax = argmax;
            }
        }
    }

    return std::make_tuple(shared_max, shared_argmax);
}


float cpu_min(Tensor *A) {
    auto t = cpu_min(A->ptr, A->size, nullptr);
    return std::get<0>(t);  // get min
}


void cpu_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    #pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        auto t = cpu_min(A->ptr, rd->index[i].size(), rd->index[i].data());
        B->ptr[i] = std::get<0>(t);  // get min
    }
}


float cpu_argmin(Tensor *A) {
    auto t = cpu_max(A->ptr, A->size, nullptr);
    return std::get<1>(t);  // get argmin
}


void cpu_argmin(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        auto t = cpu_min(A->ptr, rd->index[i].size(), rd->index[i].data());
        B->ptr[i] = std::get<1>(t);  // get argmmin
    }
}


std::tuple<float, int> cpu_min(float *ptr, int size, int *map) {
    float shared_min = MAX_FLOAT;
    int shared_argmin = 0;

#pragma omp parallel
    {
        float min = MAX_FLOAT;
        int argmin = 0;

        // TODO: I don't like this approach
        if(map == nullptr){
#pragma omp for nowait
            for (int i = 0; i < size; ++i) {
                if(ptr[i]<min){
                    min = ptr[i];
                    argmin = i;
                }
            }
        }else{
#pragma omp for nowait
            for (int i = 0; i < size; ++i) {
                if(ptr[i]<min){
                    min = ptr[map[i]];
                    argmin = i;
                }
            }
        }

#pragma omp critical
        {
            if(min<shared_argmin){
                shared_argmin = min;
                shared_argmin = argmin;
            }
        }
    }

    return std::make_tuple(shared_min, shared_argmin);
}



float cpu_sum(Tensor *A) {
    return cpu_sum(A->ptr, A->size, nullptr);
}


void cpu_sum(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    #pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_sum(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}

float cpu_sum(float *ptr, int size, int *map) {
    float sum = 0.0f;

    // TODO: I don't like this approach
    if(map == nullptr){
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; ++i) { sum += ptr[i]; }
    }else{
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; ++i) { sum += ptr[map[i]]; }
    }

    return sum;
}


float cpu_sum_abs(Tensor *A) {
    return cpu_sum_abs(A->ptr, A->size, nullptr);
}


void cpu_sum_abs(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_sum_abs(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}

float cpu_sum_abs(float *ptr, int size, int *map) {
    float sum = 0.0f;

    // TODO: I don't like this approach
    if(map == nullptr){
#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; ++i) { sum += ::fabs(ptr[i]); }
    }else{
#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; ++i) { sum += ::fabs(ptr[map[i]]); }
    }

    return sum;
}


float cpu_prod(Tensor *A) {
    return cpu_prod(A->ptr, A->size, nullptr);
}


void cpu_prod(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_prod(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}

float cpu_prod(float *ptr, int size, int *map) {
    float prod = 1.0f;

    // TODO: I don't like this approach
    if(map == nullptr){
#pragma omp parallel for reduction(*:prod)
        for (int i = 0; i < size; ++i) { prod *= ptr[i]; }
    }else{
#pragma omp parallel for reduction(*:prod)
        for (int i = 0; i < size; ++i) { prod *= ptr[map[i]]; }
    }

    return prod;
}

float cpu_mean(Tensor *A) {
    return cpu_sum(A->ptr, A->size, nullptr) / A->size;
}


void cpu_mean(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    #pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_sum(A->ptr, rd->index[i].size(), rd->index[i].data()) / rd->index[i].size();
    }
}


float cpu_var(Tensor *A, bool unbiased){
    return cpu_var(A->ptr, A->size, nullptr, unbiased);
}


void cpu_var(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_var(A->ptr, rd->index[i].size(), rd->index[i].data(), unbiased);
    }
}

float cpu_var(float *ptr, int size, int *map, bool unbiased){
    float mean = cpu_sum(ptr, size, map) / size;
    float sum = 0.0f;

    if(map == nullptr) {
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; ++i) {
            float tmp = ptr[i] - mean;
            sum += tmp * tmp;
        }
    }else{
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; ++i) {
            float tmp = ptr[map[i]] - mean;
            sum += tmp * tmp;
        }
    }
    if(unbiased){return sum/(size-1.0f);}
    else {return sum/(size);}
}


float cpu_std(Tensor *A, bool unbiased) {
    return ::sqrtf(cpu_var(A->ptr, A->size, nullptr, unbiased));
}

void cpu_std(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = ::sqrtf(cpu_var(A->ptr, rd->index[i].size(), rd->index[i].data(), unbiased));
    }
}


int cpu_mode(Tensor *A) {
    return cpu_mode(A->ptr, A->size, nullptr);
}


void cpu_mode(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_mode(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}

int cpu_mode(float *ptr, int size, int *map) {
    std::unordered_map<int, int> table;


    // TODO: I don't like this approach
    if(map == nullptr){
        // Get frequencies
        for (int i = 0; i < size; ++i){ table[(int)ptr[i]]++; }
    }else{
        // Get frequencies
        for (int i = 0; i < size; ++i){ table[(int)ptr[map[i]]]++; }
    }

    int mode = 0;
    int mode_freq = 0;
    for (auto &itor : table){
        if (itor.second > mode_freq){
            mode = itor.first;
            mode_freq = itor.second;
        }
    }

    return mode;
}


float cpu_median(Tensor *A) {
    int midpoint = A->size / 2.0f;

    if(A->size % 2==1 && A->size>1) {
        return A->ptr[midpoint];
    }else{
        return (A->ptr[midpoint-1]+A->ptr[midpoint])/2.0f;
    }
}




