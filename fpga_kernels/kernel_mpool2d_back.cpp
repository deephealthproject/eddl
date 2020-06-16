#include <math.h>
#include <stdio.h>
extern "C" {

extern void k_add_pixel(int b,int px,int py,int pz, int Dic, int Dir, int isize,int irsize,float val, float *ptr);

void k_mpool2D_back(int ir, int ic, int iz, int padrt, int padrb, int padcl, int padcr, int kr, int kc, int sr, int sc, long int size, int Ishape0, float *ptr_in, float *ptr_out, int *ptr_indX, int *ptr_indY) {

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

    for(int b=0; b < Ishape0; b++){  // Batches (ob=ib)
        int p = b * size; // Kernel's index (opt. shared variable)

        for(int k=0; k < iz; k++) { // Depth: front-back (oz=iz)
            for(int i = -padrt; i <= ir + padrb - kr; i += sr) {  // rows: top-bottom
                for(int j = -padcl; j <= ic + padcr - kc; j += sc, p++) { // cols: left-right

                    int x = ptr_indX[p];  // previous: j+kj
                    int y = ptr_indY[p];  // previous: i+ki
                    k_add_pixel(b, x, y, k, ic, ir, isize, irsize, ptr_in[p], ptr_out);  // Set input's delta

                } // cols
            } // rows
        } // depth
    } // batch
}

}
