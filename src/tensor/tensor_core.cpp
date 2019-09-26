#include "tensor.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;

void Tensor::transpose(Tensor *A, Tensor *B, vector<int> dims) {
    // Transpose
    // TODO: Review correctness
    B->tsem->lock();
    if (!Tensor::eqsize(A, B))
        msg("Tensors with different shape", "Tensor::transpose");

    if (A->device != B->device) msg("Tensors in different devices", "Tensor::transpose");

    Tensor *N;
    if (A == B) N = new Tensor(A->getShape(), A->device);
    else N = B;


    // Copy tensor data
    if (A->isCPU()) {
        for (int i = 0; i < A->size; i++)
            N->ptr[i] = A->ptr[i];
    }
#ifdef cGPU
    else if (A->isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    B->tsem->unlock();

    if (A == B) delete N;

}

void Tensor::copy(Tensor *A, Tensor *B) {
    ///////////////////////////////////////
    /// Copy from A to B
    //////////////////////////////////////
    // TODO: Review correctness for ndim==2

    if (!Tensor::eqsize(A, B)) {
        A->info();
        B->info();
        msg("Tensors with different shape", "Tensor::copy");
    }

    B->tsem->lock();
    if ((A->isCPU()) && (B->isCPU())) {
        for (int i = 0; i < A->size; i++)
            B->ptr[i] = A->ptr[i];

    }
#ifdef cGPU
        else if ((A->isGPU())&&(B->isGPU())) {
          gpu_copy_gpu(A,B);
        }
        else if ((A->isCPU())&&(B->isGPU()))
          {
            gpu_copy_to_gpu(A->ptr,B);
          }
        else if ((A->isGPU())&&(B->isCPU()))
          {
            gpu_copy_from_gpu(A,B->ptr);
          }
#endif
    else {
        fprintf(stderr, "(%d %d)\n", A->device, B->device);
        msg("unsupported copy between devices", "Tensor::copy");
    }
    B->tsem->unlock();
}

void Tensor::fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc) {
    ///////////////////////////////////////
    /// Partial copy ndim=1
    //////////////////////////////////////
    if (A->ndim != B->ndim)
        msg("Tensors with different shape", "Tensor::fill");

    B->tsem->lock();
    if ((A->isCPU()) && (B->isCPU())) {
        int at = A->size / A->shape[0];
        int bt = B->size / B->shape[0];

        int t = 1;
        for (int i = 2; i < A->ndim; i++)
            t *= A->shape[i];

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
#ifdef cGPU
        else if ((A->isGPU())&&(B->isGPU())) {
          gpu_fill(A,aini,aend,B,bini,bend,inc);
        }
#endif
    else {
        fprintf(stderr, "(%d %d)\n", A->device, B->device);
        msg("unsupported copy between devices", "Tensor::copy");
    }
    B->tsem->unlock();
}

void Tensor::select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end) {
    ///////////////////////////////////////
    /// Select from A to B
    //////////////////////////////////////

    if ((A->size / A->shape[0]) != (B->size / B->shape[0])) {
        A->info();
        B->info();
        msg("Incompatible shape", "Tensor::select");
    }

    //B->tsem->lock();
    if ((A->isCPU()) && (B->isCPU())) {
        int s = A->size / A->shape[0];

        for (int i = ini; i < end; i++) {
            int p = sind[i] * s;
            int pb = (i - ini) * s;
            for (int j = 0; j < s; j++, p++, pb++)
                B->ptr[pb] = A->ptr[p];
        }
    } else {
        msg("unsuppoted select between devices", "Tensor::select");
    }
    //B->tsem->unlock();
}
