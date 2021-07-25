/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/hardware/cpu/cpu_tensor.h" 

void cpu_repeat_nn(Tensor *A, Tensor *B, vector<int> size){
    _profile(_CPU_REPEAT_NN, 0);
#pragma omp parallel for
    for(int i=0; i<B->size; i++){
        // Get row/col of Tensor B
        int row_b = i/B->shape[2+1];  // (batch, channels, rows), cols
        int col_b = i%B->shape[2+1]; // (batch, channels, rows), cols

        // Translate row/col of Tensor B to Tensor A
        int row_a = row_b/size[0];
        int col_a = col_b/size[1];
        int offset_a = row_a*A->shape[2+1] + col_a;

        B->ptr[i] = A->ptr[offset_a];
    }
    _profile(_CPU_REPEAT_NN, 1);

}

void cpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size){
    _profile(_CPU_D_REPEAT_NN, 0);

#pragma omp parallel for
    for(int i=0; i<D->size; i++){
        // Get row/col of Tensor B
        int row_d = i/D->shape[2+1];  // (batch, channels, rows), cols
        int col_d = i%D->shape[2+1];  // (batch, channels, rows), cols

        // Translate row/col of Tensor B to Tensor A
        int row_a = row_d/size[0];
        int col_a = col_d/size[1];
        int offset_a = row_a*A->shape[2+1] + col_a;

        A->ptr[offset_a] += D->ptr[i];
    }
    _profile(_CPU_D_REPEAT_NN, 1);

}


void cpu_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
    #pragma omp parallel for
    for (int b = 0; b < B->shape[0]; b++) {
        for (int i = 0; i < B->stride[0]; i++) {
            B->ptr[b*B->stride[0] + i] = A->ptr[b*A->stride[0] + sd->cpu_addresses[i]];
        }
    }
}

void cpu_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
    #pragma omp parallel for
    for (int b = 0; b < A->shape[0]; b++) {
        for (int i = 0; i < A->stride[0]; i++) {  // walk stride
            B->ptr[b*B->stride[0] + sd->cpu_addresses[i]] += A->ptr[b*A->stride[0] + i];  // delta_parent += delta
        }
    }
}

void cpu_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
   #pragma omp parallel for
    for (int b = 0; b < B->shape[0]; b++) {
        for (int i = 0; i < B->stride[0]; i++) {
            A->ptr[b*A->stride[0] + sd->cpu_addresses[i]] = B->ptr[b*B->stride[0] + i];
        }
    }
}

void cpu_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
   #pragma omp parallel for
    for (int b = 0; b < B->shape[0]; b++) {
        for (int i = 0; i < B->stride[0]; i++) {
            B->ptr[b*B->stride[0] + i] += A->ptr[b*A->stride[0] + sd->cpu_addresses[i]];
        }
    }
}

void cpu_expand_nn(Tensor *A, Tensor *B, ExpandDescriptor *sd){
#pragma omp parallel for
    for (int b = 0; b < B->shape[0]; b++) {
        for (int i = 0; i < B->stride[0]; i++) {
            B->ptr[b*B->stride[0] + i] = A->ptr[b*A->stride[0] + sd->cpu_addresses[i]];
        }
    }
}

void cpu_expand_back_nn(Tensor *A, Tensor *B, ExpandDescriptor *sd){
#pragma omp parallel for
    for (int b = 0; b < A->shape[0]; b++) {
        for (int i = 0; i < A->stride[0]; i++) {  // walk stride
            B->ptr[b*B->stride[0] + sd->cpu_addresses[i]] += A->ptr[b*A->stride[0] + i];  // delta_parent += delta
        }
    }
}

void cpu_repeat_batch(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int b = 0; b < B->shape[0]; b++) {
        for (int i = 0; i < B->stride[0]; i++) {  // "A" must have batch of size 1
            B->ptr[b*B->stride[0] + i] = A->ptr[i];
        }
    }
}

void cpu_multithreshold(Tensor *A, Tensor *B, Tensor *thresholds, float out_bias, float out_scale) {
#ifdef CPU_DEBUG
	printf("multithreshold:\n");
	_profile_cpu_tensor(A);
	_profile_cpu_tensor(thresholds);
#endif

    if ((A->ndim == 4) && (thresholds->ndim == 2)) {

      for (int b = 0; b < A->shape[0]; b++) {
        for (int c = 0; c < A->shape[1]; c++) {
  	  for (int i = 0; i < A->shape[2] * A->shape[3]; i++) {
	    float value = A->ptr[b*A->stride[0] + c*A->stride[1] + i];
	    float p = 0;
	    for (int t = 0; t < thresholds->shape[1]; t++) {
	      float threshold;
	      if (thresholds->shape[0] == 1) threshold = thresholds->ptr[t];
	      else threshold = thresholds->ptr[c*thresholds->stride[0] + t];
	      if (value > threshold) p = p + 1.f;
	    }
//	    printf("value %f -> pos %f\n", value, p);
  	    B->ptr[b*B->stride[0] + c*B->stride[1] + i] = (p * out_scale) + out_bias;
	  }
	}
      }
    } else if ((A->ndim == 2) && (thresholds->ndim == 2)) {
      for (int b = 0; b < A->shape[0]; b++) {
        for (int i = 0; i < A->shape[1]; i++) {
          float value = A->ptr[b*A->stride[0] + i];
	  float p = 0;
	  for (int t = 0; t < thresholds->shape[1]; t++) {
	    float threshold = thresholds->ptr[t];
	    if (value > threshold) p = p + 1.f;
	  }
	  B->ptr[b*B->stride[0] + i] = (p * out_scale) + out_bias;
	}
      }
    } else {
      printf("multithreshold case not supported (A dims %d, thresholds dims %d)\n", A->ndim, thresholds->ndim);
      exit(1);
    }

#ifdef CPU_DEBUG
	_profile_cpu_tensor(B);
#endif
}

void cpu_topK(Tensor *A, Tensor *B, int axis, int largest, int sorted, int K) {
#ifdef CPU_DEBUG
	printf("topK:\n");
	printf(" input    : "); _profile_cpu_tensor(A);
#endif
	if (axis == -1) {
	  if (A->ndim == 2) {
	    if (K == 1) {
	      if (largest) {
	        for (int b = 0; b < A->shape[0]; b++) {
	          float largest = A->ptr[b*A->stride[0]];
		  int pos_largest = 0;
		  for (int i = 0; i < A->shape[1]; i++) {
		    float value = A->ptr[b*A->stride[0] + i];
		    printf("topK: value %f (pos %d)\n", value, i);
		    if (value > largest) {
		      largest = value;
		      pos_largest = i;
		    }
		  }
		  B->ptr[b*B->stride[0]] = largest;
		  printf("largest %f, position %d\n", largest, pos_largest);
		}
              } else {
	        printf("!largest not supported in topK\n"); exit(1);
              }
	    } else {
	      printf("K value not supported in topK\n"); exit(1);
	    }
	  } else {
	    printf("A tensor shape not supported in topK\n"); exit(1);
	  }
	} else {
	  printf("axis not supported in topK\n"); exit(1);
	}

#ifdef CPU_DEBUG
	printf(" output   : "); _profile_cpu_tensor(B);
#endif
}
