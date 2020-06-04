#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_RANGE
void k_range(float *A, float min, float step, int size){
   float v=min;

    for(int i=0; i<size; i++){
        A[i] = v;
        v+=step;
    }
}
#endif

#ifdef K_ENABLED_EYE
void k_eye(float *A, int offset, int size, int Ashape0, int Ashape1){
    for(int i=0; i<size; i++){
        if ((i / Ashape0 + offset) == i % Ashape1) { A[i] = 1.0f; }  // rows+offset == col?
        else { A[i] = 0.0f; }
    }
}
#endif

}
