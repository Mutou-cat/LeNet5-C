#include "rbf.h"
#include <assert.h>
#include <math.h>

/* 依据LeCun等(1998) Fig.3的7×12编码思想手工重绘0..9。
 * 不是原始源码字模的逐位复刻。空白为背景，#为笔画，逐行展平。
 * 参考来源及与原训练流程的差异见README。模板固定，不参与梯度更新。
 */
// 每行额外的第8字节存字符串结束符，展平84维时不会读取它。
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

// coordinate=k按行展平，行=k/7、列=k%7；数字模板始终只读。
float rbfCenter(int digit,int coordinate)
{
    assert(digit>=0 && digit<RBF_CLASSES && coordinate>=0 && coordinate<RBF_INPUTS);
    return digit_codes[digit][coordinate/RBF_COLS][coordinate%RBF_COLS]=='#' ? 1.0f:-1.0f;
}

// F6的84维输出与各模板求平方距离，不开平方、不加偏置、不做激活。
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

/* 令m=min(J,D[0]..D[9])，返回exp(m-J)+sum(exp(m-D[j]))。
 * 指数均<=0且至少一项为1，避免指数上溢及总和下溢为0；极小单项仍可能舍入为0。
 * log(exp(-J)+sum(exp(-D))) = log(返回值)-m。
 */
static double normalizer(const RbfLayer* layer,double* minimum)
{
    *minimum=RBF_REJECT_COST;
    for(int j=0;j<RBF_CLASSES;j++)
        if(layer->y[j]<*minimum) *minimum=layer->y[j];
    double sum=exp(*minimum-RBF_REJECT_COST);
    for(int j=0;j<RBF_CLASSES;j++) sum+=exp(*minimum-layer->y[j]);
    return sum;
}

// 当前样本MAP损失：D[target]+log(exp(-J)+sum_j exp(-D[j]))。
double rbfloss(const RbfLayer* layer,int target)
{
    assert(target>=0 && target<RBF_CLASSES);
    double minimum;
    double sum=normalizer(layer,&minimum);
    return (layer->y[target]-minimum)+log(sum);
}

/* p[j]=exp(-D[j])/(exp(-J)+sum exp(-D))，这里归一化还包含固定背景项exp(-J)。
 * dL/dD[j]=one_hot[j]-p[j]；符号与以logit为输入的交叉熵不同，因为这里是距离。
 * dD[j]/dx[k]=2*(x[k]-center[j,k])，链式法则求dx；中心固定，不更新模板。
 */
void rbfbp(RbfLayer* layer,const float input[RBF_INPUTS],int target,float dx[RBF_INPUTS])
{
    assert(target>=0 && target<RBF_CLASSES);
    double minimum;
    double sum=normalizer(layer,&minimum);
    for(int j=0;j<RBF_CLASSES;j++)
        layer->d[j]=(j==target ? 1.0:0.0)-exp(minimum-layer->y[j])/sum;
    // dx[k]=2*sum_j(dL/dD[j])*(input[k]-template[j][k])，尚未乘F6激活导数。
    for(int k=0;k<RBF_INPUTS;k++) {
        double grad=0;
        for(int j=0;j<RBF_CLASSES;j++)
            grad+=2*layer->d[j]*((double)input[k]-rbfCenter(j,k));
        dx[k]=(float)grad;
    }
}

// 选择最小距离；相等时保留较小数字编号，不执行背景类别拒识。
int rbfPredict(const RbfLayer* layer)
{
    int best=0;
    for(int j=1;j<RBF_CLASSES;j++)
        if(layer->y[j]<layer->y[best]) best=j;
    return best;
}
