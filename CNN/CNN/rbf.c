#include "rbf.h"
#include <assert.h>
#include <math.h>

/* 依据LeCun等(1998) Fig.3的7×12编码思想手工重绘0..9。
 * 不是原始源码字模的逐位复刻。空白为背景，#为笔画，逐行展平。
 * 参考来源及与原训练流程的差异见README。模板固定，不参与梯度更新。
 */
static const char digit_codes[RBF_CLASSES][RBF_ROWS][RBF_COLS+1]={
    {"       ","  ###  "," #   # "," #   # "," #   # "," #   # ",
     " #   # "," #   # "," #   # ","  ###  ","       ","       "},
    {"       ","   #   ","  ##   "," # #   ","   #   ","   #   ",
     "   #   ","   #   ","   #   "," ##### ","       ","       "},
    {"       ","  ###  "," #   # ","     # ","     # ","    #  ",
     "   #   ","  #    "," #     "," ##### ","       ","       "},
    {"       "," ##### ","     # ","    #  ","   #   ","  ###  ",
     "     # ","     # "," #   # ","  ###  ","       ","       "},
    {"       ","    #  ","   ##  ","  # #  "," #  #  "," #  #  ",
     " ##### ","    #  ","    #  ","    #  ","       ","       "},
    {"       "," ##### "," #     "," #     "," ####  ","     # ",
     "     # ","     # "," #   # ","  ###  ","       ","       "},
    {"       ","   ##  ","  #    "," #     "," #     "," ####  ",
     " #   # "," #   # "," #   # ","  ###  ","       ","       "},
    {"       "," ##### ","     # ","     # ","    #  ","   #   ",
     "   #   ","  #    ","  #    ","  #    ","       ","       "},
    {"       ","  ###  "," #   # "," #   # "," #   # ","  ###  ",
     " #   # "," #   # "," #   # ","  ###  ","       ","       "},
    {"       ","  ###  "," #   # "," #   # "," #   # ","  #### ",
     "     # ","     # ","    #  ","  ##   ","       ","       "}
};

float rbfCenter(int digit,int coordinate)
{
    assert(digit>=0 && digit<RBF_CLASSES && coordinate>=0 && coordinate<RBF_INPUTS);
    return digit_codes[digit][coordinate/RBF_COLS][coordinate%RBF_COLS]=='#' ? 1.0f:-1.0f;
}

void rbfff(RbfLayer* layer,const float input[RBF_INPUTS])
{
    for(int j=0;j<RBF_CLASSES;j++) {
        double sum=0;
        for(int k=0;k<RBF_INPUTS;k++) {
            double diff=(double)input[k]-rbfCenter(j,k);
            sum+=diff*diff;
        }
        layer->y[j]=sum;
    }
}

/* log-sum-exp：所有指数<=0，且至少有一项为1，避免下溢/上溢。 */
static double normalizer(const RbfLayer* layer,double* minimum)
{
    *minimum=RBF_REJECT_COST;
    for(int j=0;j<RBF_CLASSES;j++)
        if(layer->y[j]<*minimum) *minimum=layer->y[j];
    double sum=exp(*minimum-RBF_REJECT_COST);
    for(int j=0;j<RBF_CLASSES;j++) sum+=exp(*minimum-layer->y[j]);
    return sum;
}

double rbfloss(const RbfLayer* layer,int target)
{
    assert(target>=0 && target<RBF_CLASSES);
    double minimum;
    double sum=normalizer(layer,&minimum);
    return (layer->y[target]-minimum)+log(sum);
}

void rbfbp(RbfLayer* layer,const float input[RBF_INPUTS],int target,float dx[RBF_INPUTS])
{
    assert(target>=0 && target<RBF_CLASSES);
    double minimum;
    double sum=normalizer(layer,&minimum);
    for(int j=0;j<RBF_CLASSES;j++)
        layer->d[j]=(j==target ? 1.0:0.0)-exp(minimum-layer->y[j])/sum;
    for(int k=0;k<RBF_INPUTS;k++) {
        double grad=0;
        for(int j=0;j<RBF_CLASSES;j++)
            grad+=2*layer->d[j]*((double)input[k]-rbfCenter(j,k));
        dx[k]=(float)grad;
    }
}

int rbfPredict(const RbfLayer* layer)
{
    int best=0;
    for(int j=1;j<RBF_CLASSES;j++)
        if(layer->y[j]<layer->y[best]) best=j;
    return best;
}
