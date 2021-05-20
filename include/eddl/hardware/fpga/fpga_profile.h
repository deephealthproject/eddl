/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifndef __PROFILE
#define __PROFILE

#include "eddl/tensor/tensor.h"

#define _FPGA_ALL               0
#define _FPGA_ANY               1
#define _FPGA_ISFINITE          2
#define _FPGA_ISINF             3
#define _FPGA_ISNAN             4
#define _FPGA_ISNEGINF          5
#define _FPGA_ISPOSINF          6
#define _FPGA_LOGICAL_AND       7
#define _FPGA_LOGICAL_OR        8
#define _FPGA_LOGICAL_NOT       9
#define _FPGA_LOGICAL_XOR      10
#define _FPGA_ALLCLOSE         11
#define _FPGA_ISCLOSE          12
#define _FPGA_GREATER          13
#define _FPGA_GREATER_EQUAL    14
#define _FPGA_LESS             15
#define _FPGA_LESS_EQUAL       16
#define _FPGA_EQUAL            17
#define _FPGA_NOT_EQUAL        18
#define _FPGA_EQUAL2           19
#define _FPGA_TRANSPOSE        20
#define _FPGA_COPY             21
#define _FPGA_FILL_            22
#define _FPGA_FILL             23
#define _FPGA_SELECT           24
#define _FPGA_SELECT_BACK      25
#define _FPGA_SET_SELECT       26
#define _FPGA_SET_SELECT_BACK  27
#define _FPGA_SELECT2          28
#define _FPGA_DESELECT         29
#define _FPGA_CONCAT           30
#define _FPGA_RANGE            31
#define _FPGA_EYE              32
#define _FPGA_SINGLE_SHIFT     33
#define _FPGA_SINGLE_ROTATE    34
#define _FPGA_SINGLE_SCALE     35
#define _FPGA_SINGLE_FLIP      36
#define _FPGA_SINGLE_CROP      37
#define _FPGA_SINGLE_CROP_SCALE 38
#define _FPGA_SHIFT            39
#define _FPGA_ROTATE           40
#define _FPGA_SCALE            41
#define _FPGA_CROP             42
#define _FPGA_CROP_SCALE       43
#define _FPGA_SHIFT_RANDOM     44
#define _FPGA_ROTATE_RANDOM    45
#define _FPGA_SCALE_RANDOM     46
#define _FPGA_FLIP_RANDOM      47
#define _FPGA_CROP_RANDOM      48
#define _FPGA_CROP_SCALE_RANDOM     49
#define _FPGA_CUTOUT_RANDOM    50
#define _FPGA_RAND_UNIFORM     51
#define _FPGA_RAND_SIGNED_UNIFORM 52
#define _FPGA_BINARY           53
#define _FPGA_RAND_NORMAL      54
#define _FPGA_ABS             55
#define _FPGA_ACOS            56
#define _FPGA_ASIN            58
#define _FPGA_ATAN            59
#define _FPGA_CEIL            60
#define _FPGA_CLAMP           61
#define _FPGA_COS             62
#define _FPGA_COSH            63
#define _FPGA_FLOOR           65
#define _FPGA_INV             66
#define _FPGA_LOG             67
#define _FPGA_LOG2            68
#define _FPGA_LOG10           69
#define _FPGA_LOGN            70
#define _FPGA_MOD             71
#define _FPGA_MULT            72
#define _FPGA_NORMALIZE       73
#define _FPGA_POW             74
#define _FPGA_POWB            75
#define _FPGA_RECIPROCAL      76
#define _FPGA_REMAINDER       77
#define _FPGA_ROUND           78
#define _FPGA_RSQRT           79
#define _FPGA_SIGN            81
#define _FPGA_SIN             82
#define _FPGA_SINH            83
#define _FPGA_SQR             84
#define _FPGA_SQRT            85
#define _FPGA_TAN             86
#define _FPGA_TRUNC           88
#define _FPGA_ADD              89
#define _FPGA_INC              90
#define _FPGA_MULT2D           91
#define _FPGA_EL_DIV           92
#define _FPGA_EL_MULT          93
#define _FPGA_SIGN2            94
#define _FPGA_SUM2D_ROWWISE    95
#define _FPGA_SUM2D_COLWISE    96
#define _FPGA_MAX              97
#define _FPGA_MIN              98
#define _FPGA_SUM              99
#define _FPGA_SUM_ABS         100
#define _FPGA_REDUCE          101
#define _FPGA_REDUCE_OP       102
#define _FPGA_REDUCE_SUM2D    103
#define _FPGA_REDUCTION       104
#define _FPGA_REDUCTION_BACK  105
#define _FPGA_RELU            106
#define _FPGA_D_RELU          107
#define _FPGA_THRESHOLDED_RELU 108
#define _FPGA_D_THRESHOLDED_RELU 109
#define _FPGA_LEAKY_RELU         110
#define _FPGA_D_LEAKY_RELU       111
#define _FPGA_ELU                112
#define _FPGA_D_ELU              113
#define _FPGA_SOFTPLUS           114
#define _FPGA_D_SOFTPLUS         115
#define _FPGA_SOFTSIGN           116
#define _FPGA_D_SOFTSIGN         117
#define _FPGA_LINEAR             118
#define _FPGA_D_LINEAR           119
#define _FPGA_SIGMOID            120
#define _FPGA_D_SIGMOID          121
#define _FPGA_HARD_SIGMOID       122
#define _FPGA_D_HARD_SIGMOID     123
#define _FPGA_EXP                124
#define _FPGA_D_EXP              125
#define _FPGA_TANH               126
#define _FPGA_D_TANH             127
#define _FPGA_SOFTMAX            128
#define _FPGA_D_SOFTMAX          129
#define _FPGA_PERMUTE_CHANELS_LAST 130
#define _FPGA_PERMUTE_CHANELS_FIRST 131
#define _FPGA_PERMUTE_BATCH_LAST    132
#define _FPGA_PERMUTE_BATCH_FIRST   133
#define _FPGA_IM2COL                134
#define _FPGA_CONV2D                135
#define _FPGA_CONV2D_GRAD           136
#define _FPGA_CONV2D_BACK           137
#define _FPGA_CENT                  138
#define _FPGA_ACCURACY              139
#define _FPGA_MPOOL2D               140
#define _FPGA_MPOOL2D_BACK          141
#define _FPGA_AVGPOOL2D             142
#define _FPGA_AVGPOOL2D_BACK        143
#define _FPGA_REPEAT_NN             144
#define _FPGA_D_REPEAT_NN           145
#define _FPGA_FLIP                  146
#define _FPGA_SUM_2                 147
#define _FPGA_TRANSFORM             148
#define _NUM_FPGA_FUNCS             150
#define _FPGA_CONV2D_STM            151
#define _FPGA_CONV2D_MAXPOOL        152


extern int num_instances_fpga[_NUM_FPGA_FUNCS];
void _profile_fpga(int f_id, int end);
void _profile_fpga_tensor(Tensor *T);
void _profile_fpga_tensor_print(Tensor *T);
#endif
