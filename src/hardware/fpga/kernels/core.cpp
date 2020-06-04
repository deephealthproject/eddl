#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_FILL_
void k_fill_(float *A, float v, int size){
    for (int i = 0; i < size; ++i){
        A[i] = v;
    }
}
#endif

#ifdef K_ENABLED_FILL
void k_fill(float *A, int aini, int aend, float * B, int bini, int bend, int inc,
            int Andim, int Asize, int *Ashape, int Bsize, int Bshape0){
    int at = Asize / Ashape[0];
    int bt = Bsize / Bshape0;

    int t = 1;


    for (int i = 2; i < Andim; i++)
        t *= Ashape[i];

    for (int i = 0; i < Ashape[0]; i++) {
        int ap = (i * at) + (aini * t);
        int bp = (i * bt) + (bini * t);

        for (int j = aini; j < aend; j++) {
            for (int k = 0; k < t; k++, ap++, bp++)
                if (inc) B[bp] += A[ap];
                else B[bp] = A[ap];
        }
    }
}
#endif

#ifdef K_ENABLED_SELECT
void k_select(float *A, float *B, int *addresses, int size){
    for (int i = 0; i < size; i++) {
        B[i] = A[addresses[i]];
    }
}
#endif

#ifdef K_ENABLED_SELECT_BACK
void k_select_back(float *A, float *B, int *addresses, int size){
    for (int i = 0; i < size; i++) {  // walk stride
        B[addresses[i]] += A[i];  // delta_parent += delta
    }
}
#endif

#ifdef K_ENABLED_SET_SELECT
void k_set_select(float *A, float *B, int *addresses, int size){
    for (int i = 0; i < size; i++) {
        A[addresses[i]] = B[i];
    }
}
#endif

#ifdef K_ENABLED_SET_SELECT_BACK
void k_set_select_back(float *A, float *B, int *addresses, int size){
    for (int i = 0; i < size; i++) {
        B[i] += A[addresses[i]];
    }
}
#endif

#ifdef K_ENABLED_SELECT
void k_select(float * A, float * B, int *sind, int ini, int end,bool mask_zeros, int size, int Ashape0){
    int s = size / Ashape0;

    #pragma omp parallel for
    for (int i = ini; i < end; i++) {
        int p  = sind[i] * s;
        int pb = (i - ini) * s;
        for (int j = 0; j < s; j++, p++, pb++)
            if ((mask_zeros)&&(sind[i]==0)) B->ptr[p]=0;
            else B[pb] = A[p];
    }
}
#endif

#ifdef K_ENABLED_DESELECT
void k_deselect(float * A, float * B, int *sind, int ini, int end,int inc,bool mask_zeros, int size, int Ashape0){
    int s = size / Ashape0;

    #pragma omp parallel for
    for (int i = ini; i < end; i++) {
        int p  = sind[i] * s;
        int pb = (i - ini) * s;
        for (int j = 0; j < s; j++, p++, pb++)
            if ((mask_zeros)&&(sind[i]==0)) B[p]=0;
            else {
              if (!inc) B[p] = A[pb];
              else B[p] += A[pb];
            }
    }
}
#endif

#ifdef K_ENABLED_CONCAT
void k_concat(float *A, int AstrideAxis, int AshapeAxis, int num_tensors, float **tensors, float *sizes, int *strides, int *shapes, unsigned int axis, bool derivative){
  // Walk through all the tensors to concat one axis (once)
    unsigned int offset = 0;
    unsigned int src_stride = 0;
    int steps = AstrideAxis * AshapeAxis;  // Equivalent to A->stride[axis-1], but without the negative index problem

    // Walk through each tensor
    for (unsigned int i = 0; i < num_tensors; i++) {
        offset += src_stride;
        src_stride = strides[i] * shapes[i];

        // Copy n bytes from src to dest
        float *dest = A + offset;
        float *src = tensors[i];

        // Walk tensor i
        for (int j = 0; j < sizes[i]; j++) {
            unsigned int k = j % src_stride;  // Pos (index) in the stride (src)
            unsigned int stride_idx = j / src_stride;  // Index of the stride (src/dst)
            unsigned int dest_offset = stride_idx * steps;  // Offset in dest

            if(derivative){ src[j] += dest[dest_offset + k]; }
            else{ dest[dest_offset + k] = src[j]; }
        }
    }
}
#endif

}
