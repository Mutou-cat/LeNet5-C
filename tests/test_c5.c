#include "cnn.h"
#include <assert.h>

static void near(float a,float b,float tolerance)
{
    if(!isfinite(a)||!isfinite(b)||fabsf(a-b)>tolerance) {
        fprintf(stderr,"C5 mismatch: %.9f vs %.9f (tol %.9f)\n",a,b,tolerance);
        abort();
    }
}
static float derivative(float y)
{
    return 1.7159f*(2.0f/3.0f)*(1-(y/1.7159f)*(y/1.7159f));
}
static float loss(CNN* cnn,float** input)
{
    cnnclear(cnn);cnnff(cnn,input);
    float total=0;
    for(int j=0;j<10;j++) {
        float e=cnn->O5->y[j]-(j==3 ? 1.0f:0.0f);
        total+=0.5f*e*e;
    }
    return total;
}
int main(void)
{
    CNN cnn={0};nSize size={32,32};cnnsetup(&cnn,size,10);
    assert(cnn.layerNum==6);
    assert(cnn.C5->inputWidth==5 && cnn.C5->inputHeight==5);
    assert(cnn.C5->mapSize==5 && cnn.C5->inChannels==16);
    assert(cnn.C5->outChannels==120 && cnn.C5->isFullConnect);
    assert(cnn.O5->inputNum==120);
    CovLayer* layers[]={cnn.C1,cnn.C3,cnn.C5};
    for(int l=0;l<3;l++) {
        CovLayer* p=layers[l];
        for(int o=0;o<p->outChannels;o++) {
            p->basicData[o]=0.01f*cosf((float)o);
            for(int i=0;i<p->inChannels;i++) for(int a=0;a<5;a++) for(int b=0;b<5;b++)
                p->mapData[i][o][a][b]=covConnected(p,i,o) ?
                    0.04f*sinf((float)(1+i*3000+o*25+a*5+b)):0;
        }
    }
    for(int i=0;i<cnn.S2->outChannels;i++) cnn.S2->basicData[i]=0.03f;
    for(int i=0;i<cnn.S4->outChannels;i++) cnn.S4->basicData[i]=0.02f*(i+1);
    for(int o=0;o<10;o++) {
        cnn.O5->basicData[o]=0.01f;
        for(int k=0;k<120;k++) cnn.O5->wData[o][k]=0.05f*cosf((float)(o*120+k));
    }
    float data[32][32],*input[32],target[10]={0};target[3]=1;
    for(int r=0;r<32;r++) {
        input[r]=data[r];
        for(int c=0;c<32;c++) data[r][c]=0.3f*sinf((float)(r*32+c));
    }
    float initial_loss=loss(&cnn,input);
    /* Independent reference: reuse existing mathematical convolution for each channel. */
    nSize five={5,5};
    for(int o=0;o<120;o++) {
        float expected=0;
        for(int i=0;i<16;i++) {
            float** result=cov(cnn.C5->mapData[i][o],five,cnn.S4->y[i],five,valid);
            expected+=result[0][0];free(result[0]);free(result);
        }
        near(cnn.C5->v[o][0][0],expected,0.000003f);
        near(cnn.C5->y[o][0][0],activation_Sigma(expected,cnn.C5->basicData[o]),0.000003f);
    }
    cnnbp(&cnn,target);
    for(int o=0;o<120;o++) {
        float dy=0;
        for(int j=0;j<10;j++) dy+=cnn.O5->d[j]*cnn.O5->wData[j][o];
        near(cnn.C5->d[o][0][0],dy*derivative(cnn.C5->y[o][0][0]),0.000001f);
    }
    /* Every S4 input is connected to all 120 C5 outputs, through the flipped kernel. */
    for(int i=0;i<16;i++) for(int r=0;r<5;r++) for(int c=0;c<5;c++) {
        float dy=0;
        for(int o=0;o<120;o++) dy+=cnn.C5->d[o][0][0]*cnn.C5->mapData[i][o][4-r][4-c];
        near(cnn.S4->d[i][r][c],dy*derivative(cnn.S4->y[i][r][c]),0.000003f);
    }
    float* params[54];float grads[54],old[54];int n=0;
    /* Sample all input channels, both ends of output channels, and noncentral kernel entries. */
    for(int i=0;i<16;i++) for(int k=0;k<3;k++) {
        int o=(i*7+k*43)%120,a=(i+k)%5,b=(2*i+3*k)%5;
        params[n]=&cnn.C5->mapData[i][o][a][b];
        grads[n++]=cnn.C5->d[o][0][0]*cnn.S4->y[i][4-a][4-b];
    }
    params[n]=&cnn.C5->basicData[0];grads[n++]=cnn.C5->d[0][0][0];
    params[n]=&cnn.C5->basicData[119];grads[n++]=cnn.C5->d[119][0][0];
    params[n]=&cnn.S4->scaleData[0];grads[n++]=cnn.S4->dscaleData[0];
    params[n]=&cnn.S4->basicData[15];grads[n++]=cnn.S4->dbasicData[15];
    params[n]=&cnn.O5->wData[3][119];grads[n++]=cnn.O5->d[3]*cnn.C5->y[119][0][0];
    params[n]=&cnn.O5->basicData[3];grads[n++]=cnn.O5->d[3];
    for(int k=0;k<n;k++) {
        old[k]=*params[k];float h=0.002f;
        *params[k]=old[k]+h;float plus=loss(&cnn,input);
        *params[k]=old[k]-h;float minus=loss(&cnn,input);
        *params[k]=old[k];
        near(grads[k],(plus-minus)/(2*h),0.00008f);
    }
    near(loss(&cnn,input),initial_loss,1e-7f);
    cnnbp(&cnn,target);
    CNNOpts opts={1,0.001f};cnnapplygrads(&cnn,opts,input);
    for(int k=0;k<n;k++) near(*params[k],old[k]-opts.alpha*grads[k],0.000001f);
    assert(loss(&cnn,input)<initial_loss);
    cnnclear(&cnn);
    for(int o=0;o<120;o++) {
        near(cnn.C5->v[o][0][0],0,0);
        near(cnn.C5->y[o][0][0],0,0);
        near(cnn.C5->d[o][0][0],0,0);
    }
    puts("PASS: C5 shapes, convolution reference, input gradients, 54 finite differences, updates and clear");
}
