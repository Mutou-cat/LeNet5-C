#include "cnn.h"
#include <assert.h>

static void near(float a,float b,float tol)
{
    if(!isfinite(a) || !isfinite(b) || fabsf(a-b)>tol) {
        fprintf(stderr,"sparse check mismatch: %.8f != %.8f\n",a,b);
        abort();
    }
}
static float derivative(float y)
{
    float t=y/1.7159f;
    return 1.7159f*(2.0f/3.0f)*(1-t*t);
}
static double loss(CNN* cnn,float** x)
{
    cnnclear(cnn); cnnff(cnn,x);
    float target[10]={0};target[3]=1;
    return cnnloss(cnn,target);
}
int main(void)
{
    /* Independent readable transcription of Table I, rows are output channels. */
    const int inputs[16][6]={
        {0,1,2,-1,-1,-1},{1,2,3,-1,-1,-1},{2,3,4,-1,-1,-1},
        {3,4,5,-1,-1,-1},{0,4,5,-1,-1,-1},{0,1,5,-1,-1,-1},
        {0,1,2,3,-1,-1},{1,2,3,4,-1,-1},{2,3,4,5,-1,-1},
        {0,3,4,5,-1,-1},{0,1,4,5,-1,-1},{0,1,2,5,-1,-1},
        {0,1,3,4,-1,-1},{1,2,4,5,-1,-1},{0,2,3,5,-1,-1},
        {0,1,2,3,4,5}
    };
    bool mask[6][16]; int count=0;
    CNN cnn={0}; nSize size={32,32}; cnnsetup(&cnn,size,10);
    assert(cnn.C1->isFullConnect && !cnn.C3->isFullConnect);
    assert(cnn.C3->inputWidth==14 && cnn.C3->inputHeight==14);
    assert(cnn.C3->outChannels==16 && cnn.S4->inputWidth==10);
    assert(cnn.S4->outChannels==16 && cnn.F6->outputNum==84);
    for(int i=0;i<6;i++) for(int j=0;j<16;j++) {
        mask[i][j]=false;
        for(int k=0;k<6;k++) if(inputs[j][k]==i) mask[i][j]=true;
        assert(covConnected(cnn.C3,i,j)==mask[i][j]);
        count+=mask[i][j];
        for(int r=0;r<5;r++) for(int c=0;c<5;c++) {
            if(!mask[i][j]) near(cnn.C3->mapData[i][j][r][c],0,0);
            /* Poison disconnected slots to detect accidental use, rather than relying on zeros. */
            cnn.C3->mapData[i][j][r][c]=mask[i][j] ?
                0.07f*sinf((float)(1+i*400+j*25+r*5+c)): 9.0f;
        }
    }
    assert(count==60 && count*25+16==1516);
    for(int j=0;j<6;j++) for(int r=0;r<5;r++) for(int c=0;c<5;c++)
        cnn.C1->mapData[0][j][r][c]=0.09f*cosf((float)(j*25+r*5+c));
    for(int i=0;i<16;i++) for(int j=0;j<120;j++) for(int a=0;a<5;a++) for(int b=0;b<5;b++)
        cnn.C5->mapData[i][j][a][b]=0.04f*sinf((float)(1+i*3000+j*25+a*5+b));
    for(int j=0;j<84;j++) for(int k=0;k<120;k++)
        cnn.F6->wData[j][k]=0.06f*sinf((float)(j*120+k+1));
    float data[32][32],*x[32],target[10]={0};target[3]=1;
    for(int r=0;r<32;r++) {
        x[r]=data[r];
        for(int c=0;c<32;c++) data[r][c]=0.4f*sinf((float)(r*32+c));
    }
    loss(&cnn,x);
    /* Independent direct sparse convolution reference, not using cov/correlation. */
    for(int j=0;j<16;j++) for(int r=0;r<10;r++) for(int c=0;c<10;c++) {
        float sum=0;
        for(int i=0;i<6;i++) if(mask[i][j])
            for(int a=0;a<5;a++) for(int b=0;b<5;b++)
                sum+=cnn.C3->mapData[i][j][4-a][4-b]*cnn.S2->y[i][r+a][c+b];
        near(cnn.C3->v[j][r][c],sum,0.00001f);
    }
    cnnbp(&cnn,target);
    /* Scatter output gradients through only connected kernels to obtain dL/d(S2.y). */
    float dx[6][14][14]={0};
    for(int i=0;i<6;i++) for(int j=0;j<16;j++) if(mask[i][j])
        for(int r=0;r<10;r++) for(int c=0;c<10;c++)
            for(int a=0;a<5;a++) for(int b=0;b<5;b++)
                dx[i][r+a][c+b]+=cnn.C3->d[j][r][c]*cnn.C3->mapData[i][j][4-a][4-b];
    for(int i=0;i<6;i++) for(int r=0;r<14;r++) for(int c=0;c<14;c++)
        near(cnn.S2->d[i][r][c],dx[i][r][c]*derivative(cnn.S2->y[i][r][c]),0.00001f);
    float grads[6][16]={0},old[6][16],bias_grad[16]={0},old_bias[16];
    for(int j=0;j<16;j++) {
        old_bias[j]=cnn.C3->basicData[j];
        for(int r=0;r<10;r++) for(int c=0;c<10;c++) bias_grad[j]+=cnn.C3->d[j][r][c];
        for(int i=0;i<6;i++) {
            old[i][j]=cnn.C3->mapData[i][j][1][3];
            if(mask[i][j]) for(int r=0;r<10;r++) for(int c=0;c<10;c++)
                grads[i][j]+=cnn.C3->d[j][r][c]*cnn.S2->y[i][r+3][c+1];
        }
    }
    /* Numerical derivative for an off-center weight in each of the 96 channel pairs. */
    float h=0.005f;
    for(int i=0;i<6;i++) for(int j=0;j<16;j++) {
        float* w=&cnn.C3->mapData[i][j][1][3];
        *w=old[i][j]+h; double plus=loss(&cnn,x);
        *w=old[i][j]-h; double minus=loss(&cnn,x);
        *w=old[i][j];
        near(grads[i][j],(plus-minus)/(2*h),0.00004f);
    }
    loss(&cnn,x);cnnbp(&cnn,target);
    CNNOpts opts={1,0.1f};cnnapplygrads(&cnn,opts,x);
    for(int i=0;i<6;i++) for(int j=0;j<16;j++) {
        near(cnn.C3->mapData[i][j][1][3],old[i][j]-opts.alpha*grads[i][j],0.000001f);
        if(!mask[i][j]) for(int r=0;r<5;r++) for(int c=0;c<5;c++)
            near(cnn.C3->mapData[i][j][r][c],9.0f,0);
    }
    for(int j=0;j<16;j++) near(cnn.C3->basicData[j],old_bias[j]-opts.alpha*bias_grad[j],0.000001f);
    puts("PASS: C3 Table I, 60 connections, sparse forward/backward, 96 numerical weight checks, updates");
    return 0;
}
