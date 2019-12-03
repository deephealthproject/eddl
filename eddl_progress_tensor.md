### Development Status

| Image | Meaning |
| ------------- |------|
| ✅ | Done |
| 🔵 | In progress |
| ❌ | Todo |



---
---
# Raw-Tensor operations
---
---

Numpy-like operations over a raw-tensor object

---
## Creation ops
---


| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| zeros | ✅ | ✅ |
| ones | ✅ | ✅ |
| arange | ✅ | ✅ |
| range | ✅ | ✅ |
| linspace | ✅ | ✅ |
| logspace | ✅ | ✅ |
| eye | ✅ | ✅ |
| full | ✅ | ✅ |


---
## Indexing, Slicing, Joining, Mutating Ops
---


| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| select | ❌ | ❌ |
| cat | ❌ | ❌ |
| chunk | ❌ | ❌ |
| gather | ❌ | ❌ |
| nonzero | ❌ | ❌ |
| reshape | ✅ | ✅ |
| split | ❌ | ❌ |
| squeeze | ❌ | ❌ |
| stack | ❌ | ❌ |
| transpose | ✅ | ✅ |
| unsqueeze | ❌ | ❌ |
| where | ❌ | ❌ |
| get | ✅ | ✅ | slow
| set | ✅ | ✅ | slow


---
## Generators
---

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| bernoulli | ❌ | ❌ |
| multinomial | ❌ | ❌ |
| uniform | ✅ | ✅ |
| signed-uniform | ✅ | ✅ |
| rand normal | ✅ | ✅ |
| rand binary | ✅ | ✅ |


---
## Serialization
---

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| save | ✅ | ✅ | bin, png, bmp, tga, jpg |
| load | ✅ | ✅ | bin, png, bmp, tga, jpg, gif,... |


---
## Math operations
---

### Pointwise Ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| abs | ✅ | ✅ |
| acos | ✅ | ✅ |
| add | ✅ | ✅ |
| asin | ✅ | ✅ |
| atan | ✅ | ✅ |
| ceil | ✅ | ✅ |
| clamp | ✅ | ✅ |
| clampmax | ✅ | ✅ |
| clampmin | ✅ | ✅ |
| cos | ✅ | ✅ |
| cosh | ✅ | ✅ |
| div | ✅ | ✅ |
| exp | ✅ | ✅ |
| floor | ✅ | ✅ |
| log | ✅ | ✅ |
| log2 | ✅ | ✅ |
| log10 | ✅ | ✅ |
| logn | ✅ | ✅ |
| max* | ✅ | ❌ | Not reduced
| mean* | ❌ | ❌ | Not reduced
| median* | ❌ | ❌ | Not reduced
| min* | ✅ | ❌ | Not reduced
| mod | ✅ | ✅ |
| mode* | ✅ | ❌ | Not reduced
| mult | ✅ | ✅ |
| neg | ✅ | ✅ |
| normalize* | ✅ | ✅ | Not reduced
| pow | ✅ | ✅ |
| reciprocal | ✅ | ✅ |
| remainder | ✅ | ✅ |
| round | ✅ | ✅ |
| rsqrt | ✅ | ✅ |
| sigmoid | ✅ | ✅ |
| sign | ✅ | ✅ |
| sin | ✅ | ✅ |
| sinh | ✅ | ✅ |
| sqr | ✅ | ✅ |
| sqrt | ✅ | ✅ |
| std* | ❌ | ❌ | Not reduced
| sub | ✅ | ✅ |
| sum* | ✅ | ✅ | Not reduced by default
| tan | ✅ | ✅ |
| tanh | ✅ | ✅ |
| trunc | ✅ | ✅ |
| var* | ❌ | ❌ | Not reduced


### Reduction ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| argmax | ❌ | ❌ |
| argmin | ❌ | ❌ |
| cumprod | ❌ | ❌ |
| cumsum | ❌ | ❌ |
| max | ✅ | ✅ |
| min | ✅ | ✅ |
| mean | ✅ | ✅ |
| median | ❌ | ❌ |
| mode | ❌ | ❌ |
| norm | ❌ | ❌ |
| prod | ❌ | ❌ |
| std | ❌ | ❌ |
| sum | ❌ | ❌ |
| unique | ❌ | ❌ |
| var | ❌ | ❌ |


### Comparison ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| allclose | ❌ | ❌ |
| argsort | ❌ | ❌ |
| eq | ✅ | ❌ |
| ge | ❌ | ❌ |
| gt | ❌ | ❌ |
| isfinite | ❌ | ❌ |
| isinf | ❌ | ❌ |
| isnan | ❌ | ❌ |
| le | ❌ | ❌ |
| lt | ❌ | ❌ |
| ne | ❌ | ❌ |
| sort | ❌ | ❌ |
| topk | ❌ | ❌ |


### Other ops

| Functionality | CPU | GPU | Comments |
| ------------- |------| -----| ---------|
| cross | ❌ | ❌ |
| diag | ❌ | ❌ |
| einsum | ❌ | ❌ |
| flatten | ❌ | ❌ |
| flip | ❌ | ❌ |
| trace | ❌ | ❌ |
| dot | ❌ | ❌ |
