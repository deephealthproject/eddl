# Development Status
---

Legend:

| Functionality | Meaning |
| ------------- |------| 
| ✅        | Done | 
| 🔵         | In progress | 
| ❌         | Todo | 


## LAYERS
---

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
|  Conv2D       | 🔵  | 🔵 | Dilated, Groups...
|  Conv2DT      | ❌ | ❌ |
|  Upsampling   | ❌ | ❌ |
|  AvgPool   | ❌ | ❌ |
|  GlobalMaxPool   | ❌ | ❌ |
|  MaxPool  |  ✅ | ✅ |


### MERGE

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
|  Add    |  ✅ | ✅ |
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


### INITIALIZERS

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| Constant      | ❌ | ❌ |
| GlorotNormal  | ❌ | ❌ |
| GlorotUniform | ❌ | ❌ |
| Identity      | ❌ | ❌ |
| Orthogonal    | ❌ | ❌ |
| RandomNormal  | ❌ | ❌ |
| RandomUniform | ❌ | ❌ |


### LOSSES

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| CrossEntropy  | ✅ | ✅ |
| MSE           | ✅ | ✅ |
| SoftCE        | ✅ | ✅ |


### OPTIMIZERS

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
---

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| CategoricalAcc | ✅ | ✅ |
| MSE            | ✅ | ✅ |


## TENSOR OPERATIONS
---

Numpy-like operations

### Creation ops


| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| zeros | ✅ | ✅ |
| ones  | ✅ | ✅ |
| arange  | ✅ | ✅ |
| range  | ✅ | ✅ |
| linspace  | ✅ | ✅ |
| logspace  | ✅ | ✅ |
| eye  | ✅ | ✅ |
| full  | ✅ | ✅ |


### Indexing, Slicing, Joining, Mutating Ops

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| cat  | ❌ | ❌  |
| chunk  | ❌ | ❌  |
| gather  | ❌ | ❌  |
| nonzero  | ❌ | ❌  |
| reshape  | ✅ | ✅ |
| split  | ❌ | ❌  |
| squeeze  | ❌ | ❌  |
| stack  | ❌ | ❌  |
| transpose  | ✅ | ✅ |
| unsqueeze  | ❌ | ❌  |
| where  | ❌ | ❌  |
| get  | ❌ | ❌  |
| set  | ❌ | ❌  |


### Generators

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| bernoulli  | ❌ | ❌  |
| multinomial  | ❌ | ❌  |
| uniform | ✅ | ✅ |
| signed-uniform | ✅ | ✅ |
| normal | ✅ | ✅ |
| rand | ✅ | ✅ |
| randn | ✅ | ✅ |


### Serialization

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| save | ✅ | ✅ |
| load | ✅ | ✅ |


### Math operations

#### Pointwise Ops

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| abs | ✅ | ✅ |
| acos |  ❌ | ❌  |
| add | ✅ | ✅ |
| asin |  ❌ | ❌  |
| atan |  ❌ | ❌  |
| ceil |  ❌ | ❌  |
| clamp |  ❌ | ❌  |
| cos |  ❌ | ❌  |
| cosh |  ❌ | ❌  |
| div | ✅ | ✅ |
| exp | ✅ | ✅ |
| floor |  ❌ | ❌  |
| log | ✅ | ✅ |
| log2 | ✅ | ✅ |
| log10 | ✅ | ✅ |
| logn | ✅ | ✅ |
| mod |  ❌ | ❌  |
| mul | ✅ | ✅ |
| neg |❌ | ❌  |
| pow | ✅ | ✅ |
| reciprocal |❌ | ❌  |
| remainder |❌ | ❌  |
| round |❌ | ❌  |
| rsqrt |❌ | ❌  |
| sigmoid |❌ | ❌  |
| sign |❌ | ❌  |
| sin |  ❌ | ❌  |
| sinh |  ❌ | ❌  |
| sqr |✅ | ✅ |
| sqrt |✅ | ✅ |
| tan |  ❌ | ❌  |
| tanh |  ❌ | ❌  |
| trunc |❌ | ❌  |


#### Reduction ops

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| argmax | ❌ | ❌  |
| argmin | ❌ | ❌  |
| cumprod | ❌ | ❌  |
| cumsum | ❌ | ❌  |
| mean | ❌ | ❌  |
| median | ❌ | ❌  |
| mode | ❌ | ❌  |
| norm | ❌ | ❌  |
| prod | ❌ | ❌  |
| std | ❌ | ❌  |
| sum | ❌ | ❌  |
| unique | ❌ | ❌  |
| var | ❌ | ❌  |


#### Comparison ops

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| allclose | ❌ | ❌  |
| argsort | ❌ | ❌  |
| eq | ❌ | ❌  |
| ge | ❌ | ❌  |
| gt | ❌ | ❌  |
| isfinite | ❌ | ❌  |
| isinf | ❌ | ❌  |
| isnan | ❌ | ❌  |
| le | ❌ | ❌  |
| lt | ❌ | ❌  |
| max | ❌ | ❌  |
| min | ❌ | ❌  |
| ne | ❌ | ❌  |
| sort | ❌ | ❌  |
| topk | ❌ | ❌  |



#### Other ops

| Functionality | CPU  | GPU  | Comments |
| ------------- |------| -----| ---------|
| cross | ❌ | ❌  |
| diag | ❌ | ❌  |
| einsum | ❌ | ❌  |
| flatten | ❌ | ❌  |
| flip | ❌ | ❌  |
| trace | ❌ | ❌  |
| dot | ❌ | ❌  |