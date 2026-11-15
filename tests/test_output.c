#include "cnn.h"
#include <assert.h>
int vecmaxIndex(float* vec,int length);
static float loss(CNN* cnn,float** x)
{
    cnnclear(cnn);cnnff(cnn,x);
    float total=0;
    for(int i=0;i<10;i++) {
        float e=cnn->O5->y[i]-(i==2 ? 1.0f:0.0f);
        total+=0.5f*e*e;
    }
    return total;
}
int main(void)
{
    float negative[]={-1.5f,-1.1f,-1.4f};
    assert(vecmaxIndex(negative,3)==1);
    CNN cnn={0}; nSize size={32,32};cnnsetup(&cnn,size,10);
    float data[32][32]={0},*x[32],target[10]={0};target[2]=1;
    for(int r=0;r<32;r++) x[r]=data[r];
    for(int j=0;j<10;j++) {
        cnn.O5->basicData[j]=0.03f*(j+1);
        for(int k=0;k<cnn.O5->inputNum;k++) cnn.O5->wData[j][k]=0;
    }
    loss(&cnn,x);cnnbp(&cnn,target);
    for(int j=0;j<10;j++) {
        assert(fabsf(cnn.O5->y[j]-activation_Sigma(0,cnn.O5->basicData[j]))<1e-6f);
        float expected=cnn.O5->d[j],old=cnn.O5->basicData[j],h=0.002f;
        cnn.O5->basicData[j]=old+h;float plus=loss(&cnn,x);
        cnn.O5->basicData[j]=old-h;float minus=loss(&cnn,x);
        cnn.O5->basicData[j]=old;
        assert(fabsf(expected-(plus-minus)/(2*h))<0.0002f);
        loss(&cnn,x);cnnbp(&cnn,target);
    }
    puts("PASS: output bias finite differences and negative score argmax");
}
