# EDDL Backend Development Status
---
&nbsp;

✅: DONE

🔵: PROGRESS

❌: TODO

&nbsp;

---
# LAYERS

### CORE

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Tensor        | ✅ | ✅ |  |
| Dense         | ✅ | ✅ |  |
| Activation    | 🔵 | 🔵 | Sigmoid, LReLu ...
| BatchNorm     | ❌ | ❌ |
| Embedding     | ❌ | ❌ |
| Input         | ✅ | ✅ |  |
| Reshape       | ✅ | ✅ |  |
| Transpose     | ❌ | ❌ |
| Drop          | ✅ | ✅ |


### CONV and POOL

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Conv2D       | 🔵  | ❌ | Dilated, Groups...
|  Conv2DT      | ❌ | ❌ |
|  Upsampling   | ❌ | ❌ |
|  AvgPool   | ❌ | ❌ |
|  GlobalMaxPool   | ❌ | ❌ |
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
|  Gaussian    | ✅ | ✅ |



### OPERATORS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Abs    |  ✅ |  ❌ |
|  Diff   |  ✅ |  ✅ |
|  Div    |  ✅ |  ✅ |
|  Exp    |  ✅ |  ✅ |
|  Log    |  ✅ |  ✅ |
|  Log10    |  ✅ |  ❌|
|  Log2    |  ✅ |  ❌ |
|  Mult    |  ✅ |  ✅|
|  Pow    | ❌ | ❌ |
|  Sqrt    | ❌ | ❌ |
|  Sum    |  ✅ |  ✅ |


### REDUCTIONS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Mean    | ✅| ❌ | still test properly
|  Var    |  ✅| ❌ | still test properly
|  Sum    |  ❌| ❌ | still test properly
|  Max    |  ❌| ❌ | still test properly
|  Min    |  ❌| ❌ | still test properly

### GENERATORS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Gaussian    | ❌| ❌ | still test properly
|  Uniform    |  ❌| ❌ | still test properly

### RECURRENT


| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  LSTM    | ❌ | ❌ |
|  RNN    | ❌ | ❌ |

&nbsp;

---

# INITIALIZERS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Constant      | ❌ | ❌ |
| GlorotNormal  | ❌ | ❌ |
| GlorotUniform | ❌ | ❌ |
| Identity      | ❌ | ❌ |
| Orthogonal    | ❌ | ❌ |
| RandomNormal  | ❌ | ❌ |
| RandomUniform | ❌ | ❌ |

&nbsp;

---
## LOSSES

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| CrossEntropy  | ✅ | ✅ |
| MSE           | ✅ | ✅ |
| SoftCE        | ✅ | ✅ |

&nbsp;

---
# OPTIMIZERS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Adadelta      | ❌ | ❌ |
| Adagrad       | ❌ | ❌ |
| Adam          | ❌ | ❌ |
| Adamax        | ❌ | ❌ |
| Nadam         | ❌ | ❌ |
| RMSProp       | ❌ | ❌ |
| SGD           | ✅ | ✅ |

&nbsp;

---
# METRICS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| CategoricalAcc | ✅ | ✅ |
| MSE            | ✅ | ✅ |
