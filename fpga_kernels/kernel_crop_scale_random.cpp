#include <math.h>
#include <stdio.h>
extern "C" {

float uniform(float min, float max) {
  return 0.5;
}

void k_crop_scale_random(float *A, float *B, int Ashape0, int Ashape2, int Ashape3, int Bshape0, int Bshape1, int Bshape2, int Bshape3, float factor0, float factor1, 
		         int Astride0, int Astride1, int Astride2, int Astride3, int Bstride0, int Bstride1, int Bstride2, int Bstride3, int mode, float constant) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem

  #pragma HLS INTERFACE s_axilite port=A         bundle=control
  #pragma HLS INTERFACE s_axilite port=B         bundle=control

  #pragma HLS INTERFACE s_axilite port=Ashape0   bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape2   bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape3   bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape0   bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape1   bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape2   bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape3   bundle=control
  #pragma HLS INTERFACE s_axilite port=factor0   bundle=control
  #pragma HLS INTERFACE s_axilite port=factor1   bundle=control
  #pragma HLS INTERFACE s_axilite port=Astride0  bundle=control
  #pragma HLS INTERFACE s_axilite port=Astride1  bundle=control
  #pragma HLS INTERFACE s_axilite port=Astride2  bundle=control
  #pragma HLS INTERFACE s_axilite port=Astride3  bundle=control
  #pragma HLS INTERFACE s_axilite port=Bstride0  bundle=control
  #pragma HLS INTERFACE s_axilite port=Bstride1  bundle=control
  #pragma HLS INTERFACE s_axilite port=Bstride2  bundle=control
  #pragma HLS INTERFACE s_axilite port=Bstride3  bundle=control
  #pragma HLS INTERFACE s_axilite port=mode      bundle=control
  #pragma HLS INTERFACE s_axilite port=constant  bundle=control


  for(int b=0; b<Bshape0; b++) {

    // Compute random coordinates
    float scale = uniform(factor0, factor1);
    int h = (int)(Ashape2 * scale);
    int w = (int)(Ashape3 * scale);
    int y = (int)((Ashape2-h) * uniform(0.0f, 1.0f));
    int x = (int)((Ashape3-w) * uniform(0.0f, 1.0f));

    int coords_from_x = x;
    int coords_to_x = x+w;
    int coords_from_y = y;
    int coords_to_y = y+h;

    // single_crop_scale
    int A_hc = coords_to_x-coords_from_x+1;
    int A_wc = coords_to_y-coords_from_y+1;

    for(int c=0; c<Bshape1; c++) {
        for(int Bi=0; Bi<Bshape2; Bi++) {
            for(int Bj=0; Bj<Bshape3; Bj++) {

                if(mode==2){ // Nearest
                    // Interpolate indices
                    int Ai = (Bi * A_hc) / Bshape2 + coords_from_x;
                    int Aj = (Bj * A_wc) / Bshape3 + coords_from_y;

                    int A_pos = b*Astride0 + c*Astride1 + Ai*Astride2 + Aj*Astride3;
                    int B_pos = b*Bstride0 + c*Bstride1 + Bi*Bstride2 + Bj*Bstride3;

                    B[B_pos] = A[A_pos];
                }else{
                    printf("Mode %d not implemented", mode);
                }
            }
        }
    }
  }
}

}
