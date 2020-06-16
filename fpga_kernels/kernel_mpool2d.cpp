#include <math.h>
#include <stdio.h>
extern "C" {

extern float k_get_pixel(int b,int px,int py,int pz,int Dic, int Dir, int isize,int irsize, float *ptr);

void k_mpool2D(int ir, int ic, int iz, int padrt, int padrb, int padcl, int padcr, int kr, int kc, int sr, int sc, long int size, int Ishape0, float *ptr_in, float *ptr_out, int *ptr_indX, int *ptr_indY) {

  #pragma HLS INTERFACE m_axi port=ptr_in offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=ptr_out offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=ptr_indX offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=ptr_indY offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=ptr_in  bundle=control
  #pragma HLS INTERFACE s_axilite port=ptr_out  bundle=control
  #pragma HLS INTERFACE s_axilite port=ptr_indX bundle=control
  #pragma HLS INTERFACE s_axilite port=ptr_indY bundle=control
  #pragma HLS INTERFACE s_axilite port=ir bundle=control
  #pragma HLS INTERFACE s_axilite port=ic bundle=control
  #pragma HLS INTERFACE s_axilite port=iz bundle=control
  #pragma HLS INTERFACE s_axilite port=padrt bundle=control
  #pragma HLS INTERFACE s_axilite port=padrb bundle=control
  #pragma HLS INTERFACE s_axilite port=padcl bundle=control
  #pragma HLS INTERFACE s_axilite port=padcr bundle=control
  #pragma HLS INTERFACE s_axilite port=kr bundle=control
  #pragma HLS INTERFACE s_axilite port=kc bundle=control
  #pragma HLS INTERFACE s_axilite port=sr bundle=control
  #pragma HLS INTERFACE s_axilite port=sc bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=Ishape0 bundle=control

    int isize = ir * ic * iz;
    int irsize = ir * ic;

    for(int b=0; b<Ishape0; b++){  // Batches
        int p = b * size;  // Kernel's index (opt. shared variable)

        for(int k=0; k<iz; k++) { // Depth: front-back
            for(int i = -padrt; i <= ir + padrb - kr; i += sr) {  // rows: top-bottom
                for(int j= -padcl; j <= ic + padcr - kc; j += sc, p++) { // cols: left-right

                    // Get max value in window
                    float max = -9999.9999e-99;  // To Fix FLT_MIN; // std::numeric_limits<float>::min();
                    for(int ki = 0; ki < kr; ki++){  // rows (kernel): top-bottom
                        for(int kj = 0; kj < kc; kj++) { // cols (kernel): left-right

                            // Get value W[ki,kj] value in window
                            // ptr_in se espera
                            float v = k_get_pixel(b, j+kj, i+ki, k, ic, ir, isize, irsize, ptr_in);
                            if (v>max) {
                                max = v;
                                ptr_indX[p] = j+kj;
                                ptr_indY[p] = i+ki;
                            }

                        } // kernel cols
                    }  // kernel rows

                    // Set output value
                    ptr_out[p] = max;

                } // cols
            } // rows
        } // depth
    } // batch

}

}
