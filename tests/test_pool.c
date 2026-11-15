#include "cnn.h"
#include <assert.h>

static void near(float actual,float expected,float tolerance)
{
    if(!isfinite(actual)||!isfinite(expected)||fabsf(actual-expected)>tolerance) {
        fprintf(stderr,"gradient mismatch: %.8f versus %.8f\n",actual,expected);
        abort();
    }
}

static void free_pool(PoolLayer* p)
{
    for(int i=0;i<p->outChannels;i++) {
        for(int r=0;r<p->inputHeight/p->mapSize;r++) {
            free(p->sumData[i][r]); free(p->y[i][r]); free(p->d[i][r]);
        }
        free(p->sumData[i]); free(p->y[i]); free(p->d[i]);
    }
    free(p->sumData); free(p->y); free(p->d);
    free(p->scaleData); free(p->basicData);
    free(p->dscaleData); free(p->dbasicData); free(p);
}

/* L=sum(upstream*y): finite differences check parameter and input derivatives. */
static float pool_loss(PoolLayer* p,float*** x)
{
    poolff(p,x);
    float loss=0;
    for(int i=0;i<2;i++)
        for(int r=0;r<2;r++)
            for(int c=0;c<3;c++)
                loss+=p->y[i][r][c]*(0.1f*(1+i+r)-0.07f*c);
    return loss;
}

static void check_pool(void)
{
    PoolLayer* p=initPoolLayer(6,4,2,2,2);
    float data[2][4][6],*rows[2][4],**x[2];
    for(int i=0;i<2;i++) {
        near(p->scaleData[i],0.25f,0);
        p->scaleData[i]=i ? -0.18f:0.3f;
        p->basicData[i]=0.03f*(i+1);
        x[i]=rows[i];
        for(int r=0;r<4;r++) {
            rows[i][r]=data[i][r];
            for(int c=0;c<6;c++) data[i][r][c]=0.02f*(1+i*24+r*6+c);
        }
    }
    poolff(p,x);
    float input_grad[2][4][6],a_grad[2],b_grad[2];
    for(int i=0;i<2;i++)
        for(int r=0;r<2;r++)
            for(int c=0;c<3;c++) {
                float sum=0;
                for(int dr=0;dr<2;dr++) for(int dc=0;dc<2;dc++)
                    sum+=data[i][2*r+dr][2*c+dc];
                near(p->sumData[i][r][c],sum,1e-6f);
                near(p->y[i][r][c],activation_Sigma(p->scaleData[i]*sum,p->basicData[i]),1e-6f);
                p->d[i][r][c]=0.1f*(1+i+r)-0.07f*c;
            }
    poolbp(p);
    for(int i=0;i<2;i++) {
        a_grad[i]=p->dscaleData[i]; b_grad[i]=p->dbasicData[i];
        for(int r=0;r<4;r++) for(int c=0;c<6;c++)
            input_grad[i][r][c]=p->scaleData[i]*p->d[i][r/2][c/2];
    }
    const float h=0.001f;
    for(int i=0;i<2;i++) {
        float* params[]={&p->scaleData[i],&p->basicData[i]};
        float gradients[]={a_grad[i],b_grad[i]};
        for(int k=0;k<2;k++) {
            float old=*params[k];
            *params[k]=old+h; float plus=pool_loss(p,x);
            *params[k]=old-h; float minus=pool_loss(p,x);
            *params[k]=old;
            near(gradients[k],(plus-minus)/(2*h),0.0003f);
        }
        for(int r=0;r<4;r++) for(int c=0;c<6;c++) {
            float old=data[i][r][c];
            data[i][r][c]=old+h; float plus=pool_loss(p,x);
            data[i][r][c]=old-h; float minus=pool_loss(p,x);
            data[i][r][c]=old;
            near(input_grad[i][r][c],(plus-minus)/(2*h),0.0003f);
        }
    }
    float old_a=p->scaleData[0],old_b=p->basicData[0];
    poolapplygrads(p,0.02f);
    near(p->scaleData[0],old_a-0.02f*a_grad[0],1e-6f);
    near(p->basicData[0],old_b-0.02f*b_grad[0],1e-6f);
    free_pool(p);
}

static float network_loss(CNN* cnn,float** x,float* target)
{
    cnnclear(cnn);
    cnnff(cnn,x);
    float loss=0;
    for(int i=0;i<10;i++) {
        float e=cnn->O5->y[i]-target[i]; loss+=0.5f*e*e;
    }
    return loss;
}

int main(int argc,char** argv)
{
    assert(argc==2);
    check_pool();
    CNN cnn={0}; nSize size={32,32}; cnnsetup(&cnn,size,10);
    /* Fixed weights make this check independent of srand(time(NULL)). */
    CovLayer* convs[]={cnn.C1,cnn.C3,cnn.C5};
    int counter=0;
    for(int layer=0;layer<3;layer++) {
        CovLayer* p=convs[layer];
        for(int i=0;i<p->inChannels;i++) for(int j=0;j<p->outChannels;j++)
            for(int r=0;r<5;r++) for(int c=0;c<5;c++)
                p->mapData[i][j][r][c]=0.08f*sinf((float)++counter);
    }
    for(int i=0;i<10;i++) for(int j=0;j<cnn.O5->inputNum;j++)
        cnn.O5->wData[i][j]=0.07f*cosf((float)++counter);
    for(int i=0;i<cnn.S2->outChannels;i++) cnn.S2->basicData[i]=0.02f;
    for(int i=0;i<cnn.S4->outChannels;i++) cnn.S4->scaleData[i]=0.2f;
    for(int j=0;j<84;j++) for(int k=0;k<120;k++)
        cnn.F6->wData[j][k]=0.06f*sinf((float)(j*120+k+1));
    float data[32][32],*x[32],target[10]={0}; target[3]=1;
    for(int r=0;r<32;r++) {
        x[r]=data[r];
        for(int c=0;c<32;c++) data[r][c]=0.3f*sinf((float)(r*32+c));
    }
    network_loss(&cnn,x,target); cnnbp(&cnn,target);
    float* params[]={&cnn.S2->scaleData[0],&cnn.S2->basicData[0],
                    &cnn.S4->scaleData[0],&cnn.S4->basicData[0],
                    &cnn.C1->mapData[0][0][2][2],&cnn.C3->mapData[0][0][2][2]};
    float expected[]={cnn.S2->dscaleData[0],cnn.S2->dbasicData[0],
                      cnn.S4->dscaleData[0],cnn.S4->dbasicData[0],0,0};
    /* Check that the network update path also updates the pooling parameters. */
    float before[6]; for(int i=0;i<6;i++) before[i]=*params[i];
    CNNOpts opts={1,0.01f}; cnnapplygrads(&cnn,opts,x);
    for(int i=0;i<4;i++) near(*params[i],before[i]-opts.alpha*expected[i],1e-6f);
    /* Check derivatives again at the updated parameters. */
    network_loss(&cnn,x,target); cnnbp(&cnn,target);
    expected[0]=cnn.S2->dscaleData[0]; expected[1]=cnn.S2->dbasicData[0];
    expected[2]=cnn.S4->dscaleData[0]; expected[3]=cnn.S4->dbasicData[0];
    /* Compute selected convolution gradients directly from d and the saved inputs. */
    for(int layer=0;layer<2;layer++) {
        CovLayer* p=convs[layer]; expected[4+layer]=0;
        for(int r=0;r<p->inputHeight-4;r++) for(int c=0;c<p->inputWidth-4;c++) {
            float input=layer ? cnn.S2->y[0][r+2][c+2] : x[r+2][c+2];
            expected[4+layer]+=p->d[0][r][c]*input;
        }
    }
    const float h=0.002f;
    for(int i=0;i<6;i++) {
        float old=*params[i];
        *params[i]=old+h; float plus=network_loss(&cnn,x,target);
        *params[i]=old-h; float minus=network_loss(&cnn,x,target);
        *params[i]=old;
        near(expected[i],(plus-minus)/(2*h),0.0003f);
    }
    network_loss(&cnn,x,target);
    float saved_output[10];
    memcpy(saved_output,cnn.O5->y,sizeof(saved_output));
    savecnn(&cnn,argv[1]);
    PoolLayer* pools[]={cnn.S2,cnn.S4};
    for(int layer=0;layer<2;layer++)
        for(int i=0;i<pools[layer]->outChannels;i++) {
            pools[layer]->scaleData[i]=99;
            pools[layer]->basicData[i]=-99;
        }
    cnn.C1->mapData[0][0][0][0]=99;
    for(int i=0;i<16;i++) for(int j=0;j<120;j++)
        for(int r=0;r<5;r++) for(int c=0;c<5;c++) cnn.C5->mapData[i][j][r][c]=99;
    for(int j=0;j<120;j++) cnn.C5->basicData[j]=99;
    cnn.O5->basicData[0]=99;
    importcnn(&cnn,argv[1]);
    for(int i=0;i<cnn.C3->inChannels;i++) for(int j=0;j<cnn.C3->outChannels;j++)
        if(!covConnected(cnn.C3,i,j)) for(int r=0;r<5;r++) for(int c=0;c<5;c++)
            near(cnn.C3->mapData[i][j][r][c],0,0);
    network_loss(&cnn,x,target);
    for(int i=0;i<10;i++) near(cnn.O5->y[i],saved_output[i],1e-6f);
    /* This short process exits after the network test; legacy CNN has no destructor. */
    puts("PASS: pool forward, input/parameter gradients, updates, S2/S4 integration, model round trip");
    return 0;
}
