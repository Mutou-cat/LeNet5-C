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
// 覆盖y距离数组；input是F6的激活后输出，不是原始图片。
void rbfff(RbfLayer* layer,const float input[RBF_INPUTS]);
// 读取最近一次前向结果，target是0..9的类别编号。
double rbfloss(const RbfLayer* layer,int target);
// 覆盖d和dx；dx表示dL/d(F6.y)，要求input与前向时一致。
void rbfbp(RbfLayer* layer,const float input[RBF_INPUTS],int target,float dx[RBF_INPUTS]);
// 返回距离最小的类别；同距离时返回较小编号。
int rbfPredict(const RbfLayer* layer);

#endif
