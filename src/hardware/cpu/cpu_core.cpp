/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_hw.h"


int num_instances[_NUM_CPU_FUNCS];

void _profile_funcname(int i, char *name) {
  switch(i) {
case _CPU_ALL             : strcpy(name, "all"); break; 
case _CPU_ANY              : strcpy(name, "any"); break;
case _CPU_ISFINITE         : strcpy(name, "isfinite"); break;
case _CPU_ISINF            : strcpy(name, "isinf"); break;
case _CPU_ISNAN            : strcpy(name, "isnan"); break;
case _CPU_ISNEGINF         : strcpy(name, "isneginf"); break;
case _CPU_ISPOSINF         : strcpy(name, "isposinf"); break;
case _CPU_LOGICAL_AND      : strcpy(name, "logical_and"); break;
case _CPU_LOGICAL_OR       : strcpy(name, "logical_or"); break;
case _CPU_LOGICAL_NOT      : strcpy(name, "logical_not"); break;
case _CPU_LOGICAL_XOR      : strcpy(name, "logical_xor"); break;
case _CPU_ALLCLOSE         : strcpy(name, "allclose"); break;
case _CPU_ISCLOSE          : strcpy(name, "isclose"); break;
case _CPU_GREATER          : strcpy(name, "greater"); break;
case _CPU_GREATER_EQUAL    : strcpy(name, "greater_equal"); break;
case _CPU_LESS             : strcpy(name, "less"); break;
case _CPU_LESS_EQUAL       : strcpy(name, "less_equal"); break;
case _CPU_EQUAL            : strcpy(name, "equal"); break;
case _CPU_NOT_EQUAL        : strcpy(name, "not_equal"); break;
case _CPU_EQUAL2           : strcpy(name, "equal2"); break;
case _CPU_TRANSPOSE        : strcpy(name, "transpose"); break;
case _CPU_COPY             : strcpy(name, "copy"); break;
case _CPU_FILL_            : strcpy(name, "fill_"); break;
case _CPU_FILL             : strcpy(name, "fill"); break;
case _CPU_SELECT           : strcpy(name, "select"); break;
case _CPU_SELECT_BACK      : strcpy(name, "select_back"); break;
case _CPU_SET_SELECT       : strcpy(name, "set_select"); break;
case _CPU_SET_SELECT_BACK  : strcpy(name, "set_select_back"); break;
case _CPU_SELECT2          : strcpy(name, "select2"); break;
case _CPU_DESELECT         : strcpy(name, "deselect"); break;
case _CPU_CONCAT           : strcpy(name, "concat"); break;
case _CPU_RANGE            : strcpy(name, "range"); break;
case _CPU_EYE              : strcpy(name, "eye"); break;
case _CPU_SINGLE_SHIFT     : strcpy(name, "single_shift"); break;
case _CPU_SINGLE_ROTATE    : strcpy(name, "single_rotate"); break;
case _CPU_SINGLE_SCALE     : strcpy(name, "single_scale"); break;
case _CPU_SINGLE_FLIP      : strcpy(name, "single_flip"); break;
case _CPU_SINGLE_CROP      : strcpy(name, "single_crop"); break;
case _CPU_SINGLE_CROP_SCALE : strcpy(name, "single_crop_scale"); break;
case _CPU_SHIFT            : strcpy(name, "shift"); break;
case _CPU_ROTATE           : strcpy(name, "rotate"); break;
case _CPU_SCALE            : strcpy(name, "scale"); break;
case _CPU_CROP             : strcpy(name, "crop"); break;
case _CPU_CROP_SCALE       : strcpy(name, "crop_scale"); break;
case _CPU_SHIFT_RANDOM     : strcpy(name, "shift_random"); break;
case _CPU_ROTATE_RANDOM    : strcpy(name, "rotate_random"); break;
case _CPU_SCALE_RANDOM     : strcpy(name, "scale_random"); break;
case _CPU_FLIP_RANDOM      : strcpy(name, "flip_random"); break;
case _CPU_CROP_RANDOM      : strcpy(name, "crop_random"); break;
case _CPU_CROP_SCALE_RANDOM     : strcpy(name, "crop_scale_random"); break;
case _CPU_CUTOUT_RANDOM    : strcpy(name, "cutout_random"); break;
case _CPU_RAND_UNIFORM     : strcpy(name, "rand_uniform"); break;
case _CPU_RAND_SIGNED_UNIFORM : strcpy(name, "rand_signed_uniform"); break;
case _CPU_BINARY           : strcpy(name, "binary"); break;
case _CPU_RAND_NORMAL      : strcpy(name, "rand_normal"); break;
case _CPU_ABS_             : strcpy(name, "abs_"); break;
case _CPU_ACOS_            : strcpy(name, "acos_"); break;
case _CPU_ADD_             : strcpy(name, "add_"); break;
case _CPU_ASIN_            : strcpy(name, "asin_"); break;
case _CPU_ATAN_            : strcpy(name, "atan_"); break;
case _CPU_CEIL_            : strcpy(name, "ceil_"); break;
case _CPU_CLAMP_           : strcpy(name, "clamp_"); break;
case _CPU_COS_             : strcpy(name, "cos_"); break;
case _CPU_COSH_            : strcpy(name, "cosh_"); break;
case _CPU_EXP_             : strcpy(name, "exp_"); break;
case _CPU_FLOOR_           : strcpy(name, "floor_"); break;
case _CPU_INV_             : strcpy(name, "inv_"); break;
case _CPU_LOG_             : strcpy(name, "log_"); break;
case _CPU_LOG2_            : strcpy(name, "log2_"); break;
case _CPU_LOG10_           : strcpy(name, "log10_"); break;
case _CPU_LOGN_            : strcpy(name, "logn_"); break;
case _CPU_MOD_             : strcpy(name, "mod_"); break;
case _CPU_MULT_            : strcpy(name, "mult_"); break;
case_CPU_NORMALIZE_       : strcpy(name, "normalize_"); break;
case _CPU_POW_             : strcpy(name, "pow_"); break;
case _CPU_POWB_            : strcpy(name, "powb_"); break;
case _CPU_RECIPROCAL_      : strcpy(name, "reciprocal_"); break;
case _CPU_REMAINDER_       : strcpy(name, "remainder_"); break;
case _CPU_ROUND_           : strcpy(name, "round_"); break;
case _CPU_RSQRT_           : strcpy(name, "rsqrt_"); break;
case _CPU_SIGMOID_         : strcpy(name, "sigmoid_"); break;
case _CPU_SIGN_            : strcpy(name, "sign_"); break;
case _CPU_SIN_             : strcpy(name, "sin_"); break;
case _CPU_SINH_            : strcpy(name, "sinh_"); break;
case _CPU_SQR_             : strcpy(name, "sqr_"); break;
case _CPU_SQRT_            : strcpy(name, "sqrt_"); break;
case _CPU_TAN_             : strcpy(name, "tan_"); break;
case _CPU_TANH_            : strcpy(name, "tanh_"); break;
case _CPU_TRUNC_           : strcpy(name, "trunc_"); break;
case _CPU_ADD              : strcpy(name, "add"); break;
case _CPU_INC              : strcpy(name, "inc"); break;
case _CPU_MULT2D           : strcpy(name, "mult2D"); break;
case _CPU_EL_DIV           : strcpy(name, "el_div"); break;
case _CPU_EL_MULT          : strcpy(name, "el_mult"); break;
case _CPU_SIGN2            : strcpy(name, "sign2"); break;
case _CPU_SUM2D_ROWWISE    : strcpy(name, "sum2D_rowwise"); break;
case _CPU_SUM2D_COLWISE    : strcpy(name, "sum2D_colwise"); break;
case _CPU_MAX              : strcpy(name, "max"); break;
case _CPU_MIN              : strcpy(name, "min"); break;
case _CPU_SUM              : strcpy(name, "sum"); break;
case _CPU_SUM_ABS          : strcpy(name, "sum_abs"); break;
case _CPU_REDUCE           : strcpy(name, "reduce"); break;
case _CPU_REDUCE_OP        : strcpy(name, "reduce_op"); break;
case _CPU_REDUCE_SUM2D     : strcpy(name, "reduce_sum2D"); break;
case _CPU_REDUCTION        : strcpy(name, "reduction"); break;
case _CPU_REDUCTION_BACK   : strcpy(name, "reduction_back"); break;
case _CPU_RELU             : strcpy(name, "relu"); break;
case _CPU_D_RELU           : strcpy(name, "d_relu"); break;
case _CPU_THRESHOLDED_RELU  : strcpy(name, "thresholded_relu"); break;
case _CPU_D_THRESHOLDED_RELU  : strcpy(name, "d_thresholded_relu"); break;
case _CPU_LEAKY_RELU         : strcpy(name, "leaky_relu"); break;
case _CPU_D_LEAKY_RELU        : strcpy(name, "d_leaky_relu"); break;
case _CPU_ELU                 : strcpy(name, "elu"); break;
case _CPU_D_ELU               : strcpy(name, "d_elu"); break;
case _CPU_SOFTPLUS            : strcpy(name, "softplus"); break;
case _CPU_D_SOFTPLUS          : strcpy(name, "d_softplus"); break;
case _CPU_SOFTSIGN            : strcpy(name, "softsign"); break;
case _CPU_D_SOFTSIGN          : strcpy(name, "d_softsign"); break;
case _CPU_LINEAR              : strcpy(name, "linear"); break;
case _CPU_D_LINEAR            : strcpy(name, "d_linear"); break;
case _CPU_SIGMOID             : strcpy(name, "sigmoid"); break;
case _CPU_D_SIGMOID           : strcpy(name, "d_sigmoid"); break;
case _CPU_HARD_SIGMOID        : strcpy(name, "hard_sigmoid"); break;
case _CPU_D_HARD_SIGMOID      : strcpy(name, "d_hard_sigmoid"); break;
case _CPU_EXP                 : strcpy(name, "exp"); break;
case _CPU_D_EXP               : strcpy(name, "d_exp"); break;
case _CPU_TANH                : strcpy(name, "tanh"); break;
case _CPU_D_TANH              : strcpy(name, "d_tanh"); break;
case _CPU_SOFTMAX             : strcpy(name, "softmax"); break;
case _CPU_D_SOFTMAX           : strcpy(name, "d_softmax"); break;
case _CPU_PERMUTE_CHANELS_LAST  : strcpy(name, "permute_channels_last"); break;
case _CPU_PERMUTE_CHANELS_FIRST  : strcpy(name, "permute_channels_first"); break;
case _CPU_PERMUTE_BATCH_LAST     : strcpy(name, "permute_batch_last"); break;
case _CPU_PERMUTE_BATCH_FIRST    : strcpy(name, "permute_batch_first"); break;
case _CPU_IM2COL                 : strcpy(name, "im2col"); break;
case _CPU_CONV2D                 : strcpy(name, "conv2d"); break;
case _CPU_CONV2D_GRAD            : strcpy(name, "conv2d_grad"); break;
case _CPU_CONV2D_BACK            : strcpy(name, "conv2d_back"); break;
case _CPU_CENT                   : strcpy(name, "cent"); break;
case _CPU_ACCURACY               : strcpy(name, "accuracy"); break;
case _CPU_MPOOL2D               : strcpy(name, "mpool2d"); break;
case _CPU_MPOOL2D_BACK           : strcpy(name, "mpool2d_back"); break;
case _CPU_AVGPOOL2D              : strcpy(name, "avgpool2d"); break;
case _CPU_AVGPOOL2D_BACK         : strcpy(name, "avgpool2d_back"); break;
case _CPU_REPEAT_NN              : strcpy(name, "repeat_nn"); break;
case _CPU_D_REPEAT_NN            : strcpy(name, "d_repeat_nn"); break;
default                          : strcpy(name, "?????"); break;
}
}

void _profile(int f_id) {
  num_instances[f_id]++;
}

void _show_profile() {
  for (int i=0; i<_NUM_CPU_FUNCS; i++) {
    if (num_instances[i] != 0) {
      char func_name[50];
      _profile_funcname(i, func_name);
      printf("%50s: %d\n", func_name, num_instances[i]);  
    }
  }
}

void cpu_transpose(Tensor * A, Tensor * B) {
    _profile(_CPU_TRANSPOSE);
    #pragma omp parallel for
    for (int i = 0; i < A->size; i++){
        B->ptr[i] = A->ptr[i];
    }
}

void cpu_copy(Tensor * A, Tensor * B){
    _profile(_CPU_COPY);
    #pragma omp parallel for
    for (int i = 0; i < A->size; i++){
        B->ptr[i] = A->ptr[i];
    }
}

void cpu_fill_(Tensor *A, float v){
    _profile(_CPU_FILL_);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        A->ptr[i] = v;
    }
}

void cpu_fill(Tensor * A, int aini, int aend, Tensor * B, int bini, int bend, int inc){
    _profile(_CPU_FILL);
    int at = A->size / A->shape[0];
    int bt = B->size / B->shape[0];

    int t = 1;


    for (int i = 2; i < A->ndim; i++)
        t *= A->shape[i];

    #pragma omp parallel for
    for (int i = 0; i < A->shape[0]; i++) {
        int ap = (i * at) + (aini * t);
        int bp = (i * bt) + (bini * t);

        for (int j = aini; j < aend; j++) {
            for (int k = 0; k < t; k++, ap++, bp++)
                if (inc) B->ptr[bp] += A->ptr[ap];
                else B->ptr[bp] = A->ptr[ap];
        }
    }
}


void cpu_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile(_CPU_SELECT);
    #pragma omp parallel for
    for (int i = 0; i < B->size; i++) {
        B->ptr[i] = A->ptr[sd->cpu_addresses[i]];
    }
}

void cpu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile(_CPU_SELECT_BACK);
    #pragma omp parallel for
    for (int i = 0; i < A->size; i++) {  // walk stride
        B->ptr[sd->cpu_addresses[i]] += A->ptr[i];  // delta_parent += delta
    }
}

void cpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile(_CPU_SET_SELECT);
    #pragma omp parallel for
    for (int i = 0; i < B->size; i++) {
        A->ptr[sd->cpu_addresses[i]] = B->ptr[i];
    }
}
void cpu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile(_CPU_SET_SELECT_BACK);
    #pragma omp parallel for
    for (int i = 0; i < B->size; i++) {
        B->ptr[i] += A->ptr[sd->cpu_addresses[i]];
    }
}


void cpu_select(Tensor * A, Tensor * B, vector<int> sind, int ini, int end,bool mask_zeros){
    _profile(_CPU_SELECT2);
    int s = A->size / A->shape[0];

    #pragma omp parallel for
    for (int i = ini; i < end; i++) {
        int p  = sind[i] * s;
        int pb = (i - ini) * s;
        for (int j = 0; j < s; j++, p++, pb++)
            if ((mask_zeros)&&(sind[i]==0)) B->ptr[p]=0;
            else B->ptr[pb] = A->ptr[p];
    }
}

void cpu_deselect(Tensor * A, Tensor * B, vector<int> sind, int ini, int end,int inc,bool mask_zeros){
    _profile(_CPU_DESELECT);
    int s = A->size / A->shape[0];

    #pragma omp parallel for
    for (int i = ini; i < end; i++) {
        int p  = sind[i] * s;
        int pb = (i - ini) * s;
        for (int j = 0; j < s; j++, p++, pb++)
            if ((mask_zeros)&&(sind[i]==0)) B->ptr[p]=0;
            else {
              if (!inc) B->ptr[p] = A->ptr[pb];
              else B->ptr[p] += A->ptr[pb];
            }
    }
}

void cpu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative){
    _profile(_CPU_CONCAT);
  // Walk through all the tensors to concat one axis (once)
    unsigned int offset = 0;
    unsigned int src_stride = 0;
    int steps = A->stride[axis] * A->shape[axis];  // Equivalent to A->stride[axis-1], but without the negative index problem

    // Walk through each tensor
    for (unsigned int i = 0; i < t.size(); i++) {
        offset += src_stride;
        src_stride = t[i]->stride[axis] * t[i]->shape[axis];

        // Copy n bytes from src to dest
        float *dest = A->ptr + offset;
        float *src = t[i]->ptr;

        // Walk tensor i
        #pragma omp parallel for
        for (int j = 0; j < t[i]->size; j++) {
            unsigned int k = j % src_stride;  // Pos (index) in the stride (src)
            unsigned int stride_idx = j / src_stride;  // Index of the stride (src/dst)
            unsigned int dest_offset = stride_idx * steps;  // Offset in dest

            if(derivative){ src[j] += dest[dest_offset + k]; }
            else{ dest[dest_offset + k] = src[j]; }
        }
    }
}
