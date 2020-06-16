// In this file we enable the kernels that will be
// implemented on the FPGA

//Activations
//#define K_ENABLED_RELU
// #define K_ENABLED_D_RELU
// #define K_ENABLED_THRESHOLDED_RELU
// #define K_ENABLED_D_TRHESHOLDED_RELU
// #define K_ENABLED_LEAKY_RELU
// #define K_ENABLED_D_LEAKY_RELU
// #define K_ENABLED_ELU
// #define K_ENABLED_D_ELU
// #define K_ENABLED_SOFTPLUS
// #define K_ENABLED_D_SOFTPLUS
// #define K_ENABLED_SOFTSIGN
// #define K_ENABLED_D_SOFTSIGN
// #define K_ENABLED_LINEAR
// #define K_ENABLED_D_LINEAR
// #define K_ENABLED_SIGMOID
// #define K_ENABLED_D_SIGMOID
// #define K_ENABLED_HARD_SIGMOID
// #define K_ENABLED_D_HARD_SIGMOID
// #define K_ENABLED_EXP
// #define K_ENABLED_D_EXP
// #define K_ENABLED_TANH
// #define K_ENABLED_D_TANH
// #define K_ENABLED_SOFTMAX
// #define K_ENABLED_D_SOFTMAX

//Bn
// #define K_ENABLED_PERMUTE_CHANNELS_LAST
// #define K_ENABLED_PERMUTE_CHANNELS_FIRST
// #define K_ENABLED_PERMUTE_BATCH_LAST
// #define K_ENABLED_PERMUTE_BATCH_FIRST

//Comparison
// #define K_ENABLED_ALL
// #define K_ENABLED_ANY
// #define K_ENABLED_ISFINITE
// #define K_ENABLED_ISINF
// #define K_ENABLED_ISNAN
// #define K_ENABLED_ISNEGINF
// #define K_ENABLED_ISPOSINF
// #define K_ENABLED_LOGICAL_AND
// #define K_ENABLED_LOGICAL_OR
// #define K_ENABLED_LOGICAL_NOT
// #define K_ENABLED_LOGICAL_XOR
// #define K_ENABLED_ALLCLOSE
// #define K_ENABLED_ISCLOSE
// #define K_ENABLED_GREATER
// #define K_ENABLED_GREATER_EQUAL
// #define K_ENABLED_LESS
// #define K_ENABLED_LESS_EQUAL
// #define K_ENABLED_EQUAL
// #define K_ENABLED_NOT_EQUAL
// #define K_ENABLED_EQUAL2

//Conv
// #define K_ENABLED_IM2COL

//Core
// #define K_ENABLED_FILL_
// #define K_ENABLED_FILL
// #define K_ENABLED_SELECT
// #define K_ENABLED_SELECT_BACK
// #define K_ENABLED_SET_SELECT
// #define K_ENABLED_SET_SELECT_BACK
// #define K_ENABLED_SELECT2
// #define K_ENABLED_DESELECT
// #define K_ENABLED_CONCAT

//Create
// #define K_ENABLED_RANGE
// #define K_ENABLED_EYE

//Da
// #define K_ENABLED_SINGLE_SHIFT
// #define K_ENABLED_SINGLE_ROTATE
// #define K_ENABLED_SINGLE_SCALE
// #define K_ENABLED_SINGLE_FLIP
// #define K_ENABLED_SINGLE_CROP
// #define K_ENABLED_SINGLE_CROP_SCALE

//Generator
// #define K_ENABLED_RAND_UNIFORM
// #define K_ENABLED_RAND_SIGNED_UNIFORM
// #define K_ENABLED_RAND_BINARY
// #define K_ENABLED_RAND_NORMAL

//Losses
// #define K_ENABLED_CENT

//Math
// #define K_ENABLED_ABS_
// #define K_ENABLED_ACOS_
// #define K_ENABLED_ADD_
// #define K_ENABLED_ASIN_
// #define K_ENABLED_ATAN_
// #define K_ENABLED_CEIL_
// #define K_ENABLED_CLAMP_
// #define K_ENABLED_COS_
// #define K_ENABLED_COSH_
// #define K_ENABLED_EXP_
// #define K_ENABLED_FLOOR_
// #define K_ENABLED_INV_
// #define K_ENABLED_LOG_
// #define K_ENABLED_LOG2_
// #define K_ENABLED_LOG10_
// #define K_ENABLED_LOGN_
// #define K_ENABLED_MOD_
// #define K_ENABLED_MULT_
// #define K_ENABLED_NORMALIZE_
// #define K_ENABLED_POW_
// #define K_ENABLED_POWB_
// #define K_ENABLED_RECIPROCAL_
// #define K_ENABLED_REMAINDER_
// #define K_ENABLED_ROUND_
// #define K_ENABLED_RSQRT_
// #define K_ENABLED_SIGMOID_
// #define K_ENABLED_SIGN_
// #define K_ENABLED_SIN_
// #define K_ENABLED_SINH_
// #define K_ENABLED_SQR_
// #define K_ENABLED_SQRT_
// #define K_ENABLED_TAN_
// #define K_ENABLED_TANH_
// #define K_ENABLED_TRUNC_
// #define K_ENABLED_ADD
// #define K_ENABLED_INC
// #define K_ENABLED_EL_DIV
// #define K_ENABLED_EL_MULT
// #define K_ENABLED_SIGN2
// #define K_ENABLED_SUM2D_ROWWISE
// #define K_ENABLED_SUM2D_COLWISE
// #define K_ENABLED_MAX
// #define K_ENABLED_SUM
// #define K_ENABLED_SUM_ABS

//Metrics
// #define K_ENABLED_ACCURACY

//Pool
// #define K_ENABLED_MPOOL2D
// #define K_ENABLED_MPOOL2D_BACK
// #define K_ENABLED_AVGPOOL2D
// #define K_ENABLED_AVGPOOL2D_BACK

//Reduction
// #define K_ENABLED_REDUCE
// #define K_ENABLED_REDUCE2
// #define K_ENABLED_REDUCE_OP
// #define K_ENABLED_OPT2
#define K_ENABLED_REDUCE_SUM2D
// #define K_ENABLED_REDUCTION
// #define K_ENABLED_REDUCTION_BACK

//Tensor_nn
// #define K_ENABLED_REPEAT_NN
// #define K_ENABLED_D_REPEAT_NN
