# EDDL Backend Development Status

✅: DONE

🔵: PROGRESS

❌: TODO

## LAYERS

### CORE

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Tensor        | ✅ | ✅ |  |
| Dense         | 🔵 | 🔵 | useibas  |
| Activation    | 🔵 | 🔵 | Sigmoid, LReLu ...
| BatchNorm     | ❌ | ❌ |
| Embedding     | ❌ | ❌ |
| Input         | ✅ | ✅ |  |
| Reshape       | ✅ | ✅ |  |
| Transpose     | ❌ | ❌ |
| Drop          | 🔵 | 🔵 | minor modification


### CONV and POOL

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Conv2D       | 🔵  | ❌ | Dilated, Groups...
|  Conv2DT      | ❌ | ❌ |
|  Upsampling   | ❌ | ❌ |
|  AvgPool   | ❌ | ❌ |
|  GlobalMacPool   | ❌ | ❌ |
|  MaxPool  |  ✅ | ❌ |



### MERGE

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Add    |  ✅ | ❌ |
|  Average    | ❌ | ❌ |
|  Concat    | ❌ | ❌ |
|  MatMul    | ❌ | ❌ |
|  Max    | ❌ | ❌ |
|  Merge    | ❌ | ❌ |
|  Min    | ❌ | ❌ |
|  Substract    | ❌ | ❌ |


### NOISE

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Gaussian    | ❌ | ❌ |



### OPERATORS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Abs    |  ✅ |  ✅ |
|  Diff   |  ✅ |  ✅ |
|  Div    |  ✅ |  ✅ |
|  Exp    |  ✅ |  ✅ |
|  Log    |  ✅ |  ✅ |
|  Log10    |  ✅ |  ✅ |
|  Log2    |  ✅ |  ✅ |
|  Mean    | ❌ | ❌ |
|  Mult    |  ✅ |  ✅|
|  Pow    | ❌ | ❌ |
|  Sqrt    | ❌ | ❌ |
|  Sum    |  ✅ |  ✅ |
|  Var    | ❌ | ❌ |

### RECURRENT


| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  LSTM    | ❌ | ❌ |
|  RNN    | ❌ | ❌ |


## INITIALIZERS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Constant      | ❌ | ❌ |
| GlorotNormal  | ❌ | ❌ |
| GlorotUniform | ❌ | ❌ |
| Identity      | ❌ | ❌ |
| Orthogonal    | ❌ | ❌ |
| RandomNormal  | ❌ | ❌ |
| RandomUniform | ❌ | ❌ |


## LOSSES

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| CrossEntropy  | ✅ | ✅ |
| MSE           | ✅ | ✅ |
| SoftCE        | ✅ | ✅ |



## OPTIMIZERS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Adadelta      | ❌ | ❌ |
| Adagrad       | ❌ | ❌ |
| Adam          | ❌ | ❌ |
| Adamax        | ❌ | ❌ |
| Nadam         | ❌ | ❌ |
| RMSProp       | ❌ | ❌ |
| SGD           | ✅ | ✅ |



## METRICS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| CategoricalAcc | ✅ | ✅ |
| MSE            | ✅ | ✅ |


