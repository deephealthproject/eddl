#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "Eigen/Dense"
#include "tensor.h"

#define USEOMP


void ConvolF(Tensor *N, Tensor *K, Tensor *E,int stride, int rpad,int cpad,int threads,int batch)
{
  

  // LOWERING  N,K,E
  Tensor *I=new Tensor(K->b*K->c*K->d,E->a*E->c*E->d);
  Tensor *Kn=new Tensor(K->a, K->b*K->c*K->d);
  Tensor *O=new Tensor(K->a,E->a*E->c*E->d);

  int size=E->c*E->d;
  int kr2=K->c/2;
  int kc2=K->d/2;
  int si,sj;
  int inr=N->c;
  int inc=N->d;
  int c;
  int i,j,k,m,z,i2,j2,b,ib,im,in;
    
  if (!rpad) kr2=0;
  if (!cpad) kc2=0;

#ifdef USEOMP
#pragma omp parallel for
#endif
  for(int k=0;k<K->a;++k) {
    for(int z=0;z<K->b;++z) {
      int b=z*(K->c*K->d);
      for(int i=0;i<K->c;++i)
	for(int j=0;j<K->d;++j,++b)
	  Kn->ptr2(k,b)=K->ptr[k]->ptr[z]->ptr2(i,j);
    }
  }

#ifdef USEOMP
#pragma omp parallel for
#endif
  for(int z=0;z<K->b;++z) {
    for(int b=0;b<N->a;++b) {
      for(int i=0;i<E->c;++i) {
	int in=i*E->d;
	int ib=b*size;
	for(int j=0;j<E->d;++j,++in) {
	  int im=z*(K->c*K->d);
	  int si=(i*stride)-kr2;
	  int sj=(j*stride)-kc2;
	  for(int i2=0;i2<K->c;++i2,++si)
	    for(int j2=0,sj=(j*stride)-kc2;j2<K->d;++j2,++im,++sj)
	      if ((si<0)||(sj<0)||(si>=inr)||(sj>=inc))
		I->ptr2(im,in+ib)=0.0;
	      else
		I->ptr2(im,in+ib)=N->ptr[b]->ptr[z]->ptr2(si,sj);

	}
      }
    }
  }


  // MAKE LOWERING CONVOLUTION
  O->ptr2=Kn->ptr2*I->ptr2;

  // RESHAPE
#ifdef USEOMP
#pragma omp parallel for
#endif
  for(int k=0;k<E->b;++k) {
    for(int b=0;b<E->a;++b) {
      int ib=b*size;
      int z=0;
      for(int i=0;i<E->c;++i)
	for(int j=0;j<E->d;++j,++z) {
	  E->ptr[b]->ptr[k]->ptr2(i,j)=O->ptr2(k,z+ib);
	}
    }
  }
    
  
  delete I;
  delete Kn;
  delete O;

  
}


///////////////////
// BACKWARD
///////////////////
void ConvolBGrad(Tensor *N, Tensor *gK, Tensor *D,int stride, int rpad,int cpad,int threads,int batch)
{

    int kr2=gK->c/2;
    int kc2=gK->d/2;
    int inr=N->c;
    int inc=N->d;

    if (!rpad) kr2=0;
    if (!cpad) kc2=0;
    
    // LOWERING N,gK,D
    Tensor *I=new Tensor(gK->b*gK->c*gK->d,D->a*D->c*D->d);
    Tensor *Dn=new Tensor(D->a*D->c*D->d,gK->a);
    Tensor *Res=new Tensor(D->a*D->c*D->d,gK->a);

    // Prepare I

#ifdef USEOMP
#pragma omp parallel for
#endif
    for(int z=0;z<gK->b;++z) {
      int q=z*(gK->c*gK->d);
      for(int i=0;i<gK->c;++i)
	for(int j=0;j<gK->d;++j,++q) {
	  int p=0;
	  for(int b=0;b<D->a;++b) {
	    for(int i2=0;i2<D->c;++i2)
	      for(int j2=0;j2<D->d;++j2,++p) {
		int si=((i2*stride)-kr2)+i;
		int sj=((j2*stride)-kc2)+j;
		if ((si<0)||(si>=inr)||(sj<0)||(sj>=inc)) I->ptr2(q,p)=0.0;
		else I->ptr2(q,p)=N->ptr[b]->ptr[z]->ptr2(si,sj);
	      }
	  }
	}
    }
    
    //Prepare Delta
#ifdef USEOMP
#pragma omp parallel for
#endif
    for(int k=0;k<gK->a;++k) {
      int p=0;
      for(int b=0;b<D->a;++b)
	for(int i=0;i<D->c;++i)
	  for(int j=0;j<D->d;++j,++p)
	    Dn->ptr2(p,k)=D->ptr[b]->ptr[k]->ptr2(i,j);
    }

    Res->ptr2=I->ptr2*Dn->ptr2;

    // Reshape to gradient
#ifdef USEOMP
#pragma omp parallel for
#endif
    for(int k=0;k<gK->a;++k) {
      int p=0;
      for(int z=0;z<gK->b;++z) {
	for(int i2=0;i2<gK->c;++i2)
	  for(int j2=0;j2<gK->d;++j2,++p)
	    gK->ptr[k]->ptr[z]->ptr2(i2,j2)=Res->ptr2(p,k);
      }
    }

    delete I;
    delete Dn;
    delete Res;
 
}



void ConvolBDelta(Tensor *D, Tensor *K, Tensor *ID,int stride, int rpad,int cpad,int threads,int batch)
{
 
  // LOWERING D K ID

  Tensor *Del=new Tensor(D->c,D->d,D->a,K->a);
  Tensor *Kr=new Tensor(K->b,K->a,K->c*K->d);
  Tensor *Res=new Tensor(D->c,D->d,K->a,K->c*K->d);

  int kr2=K->c/2;
  int kc2=K->d/2;
  int inr=ID->c;
  int inc=ID->d;
    
  if (!rpad) kr2=0;
  if (!cpad) kc2=0;

#ifdef USEOMP
#pragma omp parallel for
#endif
  for(int z=0;z<K->b;++z) {
    for(int k=0;k<K->a;++k)
      for(int i2=0,r=0,c=0;i2<K->c*K->d;++i2,++c) {
	if (c==K->d) {c=0;r++;}
	Kr->ptr[z]->ptr2(k,i2)=K->ptr[k]->ptr[z]->ptr2(r,c);
      }
  }



#ifdef USEOMP
#pragma omp parallel for
#endif
  for(int i=0;i<D->c;++i) {
    for(int j=0;j<D->d;++j) {
      for(int b=0;b<D->a;++b)
	for(int k=0;k<K->a;++k)
	  Del->ptr[i]->ptr[j]->ptr2(b,k)=D->ptr[b]->ptr[k]->ptr2(i,j);
    }
  }

#ifdef USEOMP
  setNbThreads(1);
#pragma omp parallel for
#endif
  for(int i=0;i<D->c;++i) {
    for(int j=0;j<D->d;++j) {
      for(int z=0;z<K->b;++z) {
	Res->ptr[i]->ptr[j]->ptr2=Del->ptr[i]->ptr[j]->ptr2*Kr->ptr[z]->ptr2;

	for(int b=0;b<D->a;++b) {
	  for(int i2=0,r=0,c=0;i2<K->c*K->d;++i2,++c) {
	    if (c==K->d) {c=0;r++;}
	    int si=(i*stride+r)-kr2;
	    int sj=(j*stride+c)-kc2;
	    if ((si<0)||(sj<0)||(si>=inr)||(sj>=inc)) { }
	    else ID->ptr[b]->ptr[z]->ptr2(si,sj)+=Res->ptr[i]->ptr[j]->ptr2(b,i2);
	  }
	}

      }//z

    }//j
  }//i

#ifdef USEOMP
  setNbThreads(threads);
#endif
  
  delete Del;
  delete Kr;
  delete Res;
}

