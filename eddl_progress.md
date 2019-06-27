# EDDL Backend Development Status

## LAYERS

### CORE

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Tensor        | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |  |
| Dense         | <span style="color:blue">**PROGRESS**</span> | <span style="color:blue">**PROGRESS**</span> | useibas  |
| Activation    | <span style="color:blue">**PROGRESS**</span> | <span style="color:blue">**PROGRESS**</span> | Sigmoid, LReLu ...
| BatchNorm     | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Embedding     | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Input         | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |  |
| Reshape       | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |  |
| Transpose     | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Drop          | <span style="color:blue">**PROGRESS**</span> | <span style="color:blue">**PROGRESS**</span> | minor modification



### CONV and POOL

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Conv2D       | <span style="color:blue">**PROGRESS**</span>  | <span style="color:red">**TODO**</span> | Dilated, Groups...
|  Conv2DT      | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Upsampling   | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  AvgPool   | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  GlobalMacPool   | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  MaxPool  |  <span style="color:green">**DONE**</span> | <span style="color:red">**TODO**</span> |



### MERGE

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Add    |  <span style="color:green">**DONE**</span> | <span style="color:red">**TODO**</span> |
|  Average    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Concat    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  MatMul    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Max    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Merge    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Min    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Substract    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |


### NOISE

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Gaussian    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |



### OPERATORS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Abs    |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span> |
|  Diff   |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span> |
|  Div    |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span> |
|  Exp    |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span> |
|  Log    |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span> |
|  Log10    |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span> |
|  Log2    |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span> |
|  Mean    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Mult    |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span>|
|  Pow    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Sqrt    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  Sum    |  <span style="color:green">**DONE**</span> |  <span style="color:green">**DONE**</span> |
|  Var    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |

### RECURRENT


| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  LSTM    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
|  RNN    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |


## INITIALIZERS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Constant      | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| GlorotNormal  | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| GlorotUniform | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Identity      | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Orthogonal    | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| RandomNormal  | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| RandomUniform | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |


## LOSSES

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| CrossEntropy  | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |
| MSE           | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |
| SoftCE        | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |



## OPTIMIZERS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Adadelta      | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Adagrad       | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Adam          | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Adamax        | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| Nadam         | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| RMSProp       | <span style="color:red">**TODO**</span> | <span style="color:red">**TODO**</span> |
| SGD           | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |



## METRICS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| CategoricalAcc | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |
| MSE            | <span style="color:green">**DONE**</span> | <span style="color:green">**DONE**</span> |


