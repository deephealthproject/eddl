#include <math.h>
#include <stdio.h>
extern "C" {

float k_get_pixel(int b,int px,int py,int pz,int Dic, int Dir, int isize,int irsize, float *ptr) {
  // Check boundaries of the window
  if (px<0) return 0.0;
  if (py<0) return 0.0;
  if (px>=Dic) return 0.0;
  if (py>=Dir) return 0.0;

  // Compute address from indices (row-major)
  unsigned int address = (b*isize) + (pz*irsize) + (py*Dic) + px;
  return ptr[address];
}

void k_add_pixel(int b,int px,int py,int pz, int Dic, int Dir, int isize,int irsize,float val, float *ptr) {
  // Check boundaries of the window
  if (px<0) return;
  if (py<0) return;
  if (px>=Dic) return;
  if (py>=Dir) return;

  // Compute address from indices (row-major)
  unsigned int address = (b*isize) + (pz*irsize) + (py*Dic) + px;
  ptr[address]+=val;
}

#ifdef K_ENABLED_IM2COL
void im2col(int b, int Dkr, int Dkc, int Dr, int Dc, int Dir, int Dic, int Diz, int Dsr, int Dsc, int Dpadrt, int Dpadcl, int matIrows, int matIcols, float *ptrI,int col2im, unsigned int *ptraddr)

{
  int i,j,k;
  int pz,py,px,y,x;
  int ksize=Dkr*Dkc;
  int kr2=Dkr/2;
  int kc2=Dkc/2;

  if (kc2==0) kc2=-1;

  int orsize=Dr*Dc;

  int isize=Dir*Dic*Diz;
  int irsize=Dir*Dic;

  k=0;
  py=-Dpadrt;
  px=-Dpadcl;


  for(j=0;j<matIrows;j++) {
    k=j;

    for(i=0;i<matIcols;i++,k+=orsize) {
      pz=i/ksize;
      y=py+(i%ksize)/Dkc;
      x=px+(i%Dkc);

      //if(col2im)
          //k_add_pixel(b,x,y,pz, Dic, Dir,isize,irsize,ptrI[k], ptraddr);
      //else
         //ptrI[k] = k_get_pixel(b,x,y,pz,Dic, Dir,isize,irsize, ptraddr);

    }
    px+=Dsc;
    if (px>=Dic+Dpadcl-kc2-1) {
      px=-Dpadcl;
      py+=Dsr;
    }
  }
}
#endif

// convolutional functions are not implemented as separate kernels. All convolution operations
// are translated to im2col operation, matmul, and matrix additions. Each channel is
// processesed separately with a loop running on the CPU (see src/hardware/fpga/nn/fpga_conv.cpp)

}
