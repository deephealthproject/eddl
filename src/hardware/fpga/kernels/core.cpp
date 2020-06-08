#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_FILL_
void k_fill_(float *A, float v, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=v bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; ++i){
        A[i] = v;
    }
}
#endif

#ifdef K_ENABLED_FILL
void k_fill(float *A, int aini, int aend, float * B, int bini, int bend, int inc,
            int Andim, int Asize, int *Ashape, int Bsize, int Bshape0){

    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=A  bundle=control
    #pragma HLS INTERFACE m_axi port=Ashape offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=Ashape  bundle=control
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=B  bundle=control
    #pragma HLS INTERFACE s_axilite port=aini bundle=control
    #pragma HLS INTERFACE s_axilite port=aend bundle=control
    #pragma HLS INTERFACE s_axilite port=bini bundle=control
    #pragma HLS INTERFACE s_axilite port=bend bundle=control
    #pragma HLS INTERFACE s_axilite port=inc bundle=control
    #pragma HLS INTERFACE s_axilite port=Andim bundle=control
    #pragma HLS INTERFACE s_axilite port=Asize bundle=control
    #pragma HLS INTERFACE s_axilite port=Bsize bundle=control
    #pragma HLS INTERFACE s_axilite port=Bshape0 bundle=control

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
void k_select(float *A, float *B, int *addresses, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=addresses offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=addresses  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {
        B[i] = A[addresses[i]];
    }
}
#endif

#ifdef K_ENABLED_SELECT_BACK
void k_select_back(float *A, float *B, int *addresses, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=addresses offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=addresses  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {  // walk stride
        B[addresses[i]] += A[i];  // delta_parent += delta
    }
}
#endif

#ifdef K_ENABLED_SET_SELECT
void k_set_select(float *A, float *B, int *addresses, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=addresses offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=addresses  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {
        A[addresses[i]] = B[i];
    }
}
#endif

#ifdef K_ENABLED_SET_SELECT_BACK
void k_set_select_back(float *A, float *B, int *addresses, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=addresses offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=addresses  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {
        B[i] += A[addresses[i]];
    }
}
#endif

#ifdef K_ENABLED_SELECT
void k_select2(float * A, float * B, int *sind, int ini, int end, bool mask_zeros, long int size, int Ashape0){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=sind offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=sind  bundle=control
  #pragma HLS INTERFACE s_axilite port=ini bundle=control
  #pragma HLS INTERFACE s_axilite port=end bundle=control
  #pragma HLS INTERFACE s_axilite port=mask_zeros bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control

    int s = size / Ashape0;

    #pragma omp parallel for
    for (int i = ini; i < end; i++) {
        int p  = sind[i] * s;
        int pb = (i - ini) * s;
        for (int j = 0; j < s; j++, p++, pb++)
            if ((mask_zeros)&&(sind[i]==0)) B[p]=0;
            else B[pb] = A[p];
    }
}
#endif

#ifdef K_ENABLED_DESELECT
void k_deselect(float * A, float * B, int *sind, int ini, int end, int inc, bool mask_zeros, long int size, int Ashape0){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=sind offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=sind  bundle=control
  #pragma HLS INTERFACE s_axilite port=ini bundle=control
  #pragma HLS INTERFACE s_axilite port=end bundle=control
  #pragma HLS INTERFACE s_axilite port=mask_zeros bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control

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

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=tensors offset=slave bundle=gmem //to check
  #pragma HLS INTERFACE m_axi port=sizes offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=strides offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=shapes offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=tensors  bundle=control
  #pragma HLS INTERFACE s_axilite port=sizes  bundle=control
  #pragma HLS INTERFACE s_axilite port=strides bundle=control
  #pragma HLS INTERFACE s_axilite port=shapes bundle=control
  #pragma HLS INTERFACE s_axilite port=AstrideAxis bundle=control
  #pragma HLS INTERFACE s_axilite port=AshapeAxis bundle=control
  #pragma HLS INTERFACE s_axilite port=num_tensors bundle=control
  #pragma HLS INTERFACE s_axilite port=axis bundle=control
  #pragma HLS INTERFACE s_axilite port=derivative bundle=control
  
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
