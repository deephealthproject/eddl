/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

// GPU: Truth value testing
__global__ void gpu_logical_all(float *A, int size, bool &result){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    // if(!result) return;  // Abort if there is a result

    if (thread_id_x < size && result){
        if (A[thread_id_x] != 1.0f){
            result = false;
            // return;
        }
    }
}

__global__ void gpu_logical_any(float *A, int size, bool &result){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    // if(result) return;  // Abort if there is a result

    if (thread_id_x < size && !result){
        if (A[thread_id_x] == 1.0f){
            result = true;
            // return;
        }
    }
}

__global__ void gpu_isfinite(float *A, float *B, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = isfinite(A[thread_id_x]);
    }
}

__global__ void gpu_isinf(float *A, float *B, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = isinf(A[thread_id_x]);
    }
}

__global__ void gpu_isnan(float *A, float *B, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = isnan(A[thread_id_x]);
    }
}

__global__ void gpu_isneginf(float *A, float *B, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = isinf(A[thread_id_x]) && A[thread_id_x] < 0.0f;
    }
}

__global__ void gpu_isposinf(float *A, float *B, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = isinf(A[thread_id_x]) && A[thread_id_x] > 0.0f;
    }
}


__global__ void gpu_logical_and(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = (bool)A[thread_id_x] & (bool)B[thread_id_x];
    }
}

__global__ void gpu_logical_or(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = (bool)A[thread_id_x] | (bool)B[thread_id_x];
    }
}

__global__ void gpu_logical_not(float *A, float *B, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = !((bool)A[thread_id_x]);
    }
}

__global__ void gpu_logical_xor(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = (bool)A[thread_id_x] ^ (bool)B[thread_id_x];
    }
}


__global__  void gpu_logical_allclose(float *A, float *B, float rtol, float atol, bool equal_nan, int size, bool &allclose){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    // if(!allclose) return;  // Abort if there is a result

    if (thread_id_x < size && allclose){
        bool close = fabsf(A[thread_id_x] - B[thread_id_x]) <= (atol + rtol * fabsf(B[thread_id_x]));
        if (!close){
            allclose = false;
            // return;
        }
    }
}

__global__  void gpu_logical_isclose(float *A, float *B, float *C, float rtol, float atol, bool equal_nan, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = fabsf(A[thread_id_x] - B[thread_id_x]) <= (atol + rtol * fabsf(B[thread_id_x]));
    }
}

__global__  void gpu_logical_greater(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] > B[thread_id_x];
    }
}

__global__  void gpu_logical_greater_equal(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] >= B[thread_id_x];
    }
}

__global__  void gpu_logical_less(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] < B[thread_id_x];
    }
}

__global__  void gpu_logical_less_equal(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] <= B[thread_id_x];
    }
}

__global__  void gpu_logical_equal(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] == B[thread_id_x];
    }
}

__global__  void gpu_logical_not_equal(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] != B[thread_id_x];
    }
}
