#include "tensor.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif


using namespace std;


void Tensor::save(string fname) {
    if (!isCPU())
        msg("Only save CPU Tensors", "Tensor::save");

    int i, j;
    FILE *fe;
    float fv;

    fe = fopen(fname.c_str(), "wb");
    if (fe == nullptr) {
        fprintf(stderr, "Not abel to write %s \n", fname.c_str());
        exit(1);
    }

    fprintf(stderr, "writting bin file\n");

    fwrite(&ndim, sizeof(int), 1, fe);
    for (i = 0; i < ndim; ++i)
        fwrite(&shape[i], sizeof(int), 1, fe);

    fwrite(ptr, sizeof(float), size, fe);

    fclose(fe);
}

void Tensor::save(FILE *fe) {
    if (!isCPU())
        msg("Only save CPU Tensors", "Tensor::save");

    fwrite(ptr, sizeof(float), size, fe);
}

void Tensor::load(FILE *fe) {
    if (!isCPU())
        msg("Only save CPU Tensors", "Tensor::save");

    fread(ptr, sizeof(float), size, fe);
}
