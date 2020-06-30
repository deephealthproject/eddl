#include "libspir_types.h"
#include "hls_stream.h"
#include "xcl_top_defines.h"
#include "ap_axi_sdata.h"
#define EXPORT_PIPE_SYMBOLS 1
#include "cpu_pipes.h"
#undef EXPORT_PIPE_SYMBOLS
#include "xcl_half.h"
#include <cstddef>
#include <vector>
#include <complex>
#include <pthread.h>
using namespace std;

extern "C" {

void k_relu(size_t A, size_t B, unsigned long size);

static pthread_mutex_t __xlnx_cl_k_relu_mutex = PTHREAD_MUTEX_INITIALIZER;
void __stub____xlnx_cl_k_relu(char **argv) {
  void **args = (void **)argv;
  size_t A = *((size_t*)args[0+1]);
  size_t B = *((size_t*)args[1+1]);
  unsigned long size = *((unsigned long*)args[2+1]);
 pthread_mutex_lock(&__xlnx_cl_k_relu_mutex);
  k_relu(A, B, size);
  pthread_mutex_unlock(&__xlnx_cl_k_relu_mutex);
}
}
