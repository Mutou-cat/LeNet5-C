#include "rbf.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void near(double a,double b,double tol)
{
    if(!isfinite(a)||!isfinite(b)||fabs(a-b)>tol) {
        fprintf(stderr,"RBF mismatch %.12g vs %.12g\n",a,b);abort();
    }
}
int main(void)
{
    RbfLayer layer={0};float x[84],dx[84];
    for(int t=0;t<10;t++) {
        for(int k=0;k<84;k++) {
            x[k]=rbfCenter(t,k);
            assert(x[k]==1 || x[k]==-1);
        }
        rbfff(&layer,x);near(layer.y[t],0,0);assert(rbfPredict(&layer)==t);
        for(int j=0;j<10;j++) {
            int mismatches=0;
            for(int k=0;k<84;k++) mismatches+=x[k]!=rbfCenter(j,k);
            near(layer.y[j],4*mismatches,0);
            if(j!=t) assert(mismatches>0);
        }
        /* Near a template gives competition, unlike only testing large distances. */
        for(int k=0;k<84;k++) x[k]*=0.75f;
        rbfff(&layer,x);
        double denom=exp(-RBF_REJECT_COST);
        for(int j=0;j<10;j++) denom+=exp(-layer.y[j]);
        near(rbfloss(&layer,t),layer.y[t]+log(denom),1e-12);
        rbfbp(&layer,x,t,dx);
        for(int j=0;j<10;j++) near(layer.d[j],(j==t)-exp(-layer.y[j])/denom,1e-12);
        for(int k=0;k<84;k++) {
            float old=x[k],h=0.001f;
            x[k]=old+h;float upper=x[k];rbfff(&layer,x);double plus=rbfloss(&layer,t);
            x[k]=old-h;float lower=x[k];rbfff(&layer,x);double minus=rbfloss(&layer,t);
            x[k]=old;
            near(dx[k],(plus-minus)/(upper-lower),0.00001);
        }
    }
    /* Equal distances choose the first class, and gradients remain finite. */
    for(int k=0;k<84;k++) x[k]=0;
    rbfff(&layer,x);assert(rbfPredict(&layer)==0);
    for(int j=0;j<10;j++) near(layer.y[j],84,0);
    near(rbfloss(&layer,4),74,1e-12);
    rbfbp(&layer,x,4,dx);
    for(int k=0;k<84;k++) near(dx[k],-2*rbfCenter(4,k),1e-6);
    for(int j=0;j<10;j++) layer.y[j]=1e6+j;
    near(rbfloss(&layer,3),1e6+3-RBF_REJECT_COST,1e-9);
    rbfbp(&layer,x,3,dx);
    for(int k=0;k<84;k++) assert(isfinite(dx[k]));
    puts("PASS: fixed templates, squared distances, argmin, stable MAP loss and 840 input gradients");
}
