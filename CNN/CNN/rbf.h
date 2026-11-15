#ifndef DLC_RBF_H
#define DLC_RBF_H

enum { RBF_INPUTS=84, RBF_CLASSES=10, RBF_ROWS=12, RBF_COLS=7 };
/* 论文式(9)中的背景类别惩罚j；10是本实现的明确选择。 */
#define RBF_REJECT_COST 10.0

typedef struct rbf_layer {
    double y[RBF_CLASSES]; /* 平方欧氏距离，越小越匹配，不是概率 */
    double d[RBF_CLASSES]; /* dL/d距离；此层没有激活函数或可训练参数 */
} RbfLayer;

/* 固定7×12数字模板，按行展平；笔画+1，背景-1。 */
float rbfCenter(int digit,int coordinate);
void rbfff(RbfLayer* layer,const float input[RBF_INPUTS]);
double rbfloss(const RbfLayer* layer,int target);
void rbfbp(RbfLayer* layer,const float input[RBF_INPUTS],int target,float dx[RBF_INPUTS]);
int rbfPredict(const RbfLayer* layer);

#endif
