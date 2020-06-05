#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_ACCURACY
void k_accuracy(float *A, float *B, int Ashape0, int Ashape1, int *accuracy) {

  int acc = 0;
  int aind = 0;
  int bind = 0;
  float maxA = 0.f;
  float maxB = 0.f;

  for (int i = 0; i < Ashape0; i++) {

    aind = 0;
    maxA = A[ Ashape1 * i];
    bind = 0;
    maxB = B[ Ashape1 * i];
    for (int j=0; j< Ashape1; j++) {
      int pos = Ashape1 * i + j;
      float valueA = A[pos];
      if (valueA > maxA) {
        aind = j;
        maxA = valueA;
      }
      float valueB = B[pos];
      if (valueB > maxB) {
        bind = j;
        maxB = valueB;
      }
    }
    if (aind == bind) acc++;
  }
  *accuracy = acc;
}
#endif

}
