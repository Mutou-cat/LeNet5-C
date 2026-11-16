#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "mat.h"
#include "minst.h"
#include "rbf.h"


// 卷积层
typedef struct convolutional_layer{
	int inputWidth;   // 输入特征图宽度
	int inputHeight;  // 输入特征图高度
	int mapSize;      // 正方形卷积核的边长

	int inChannels;   // 输入通道数，不是样本数
	int outChannels;  // 输出通道数，不是样本数

	// 四维权重按[输入通道][输出通道][行][列]寻址。
	// 部分连接保留完整存储槽；无效槽不参与前向、反向和更新。
	float**** mapData;     // W[输入通道][输出通道][核行][核列]
	float**** dmapData;    // 与W同形；当前仅C5缓存dL/dW，C1/C3使用临时矩阵

	float* basicData;   // 每输出通道一个偏置，空间位置之间共享
	bool isFullConnect; // 是否连接全部输入通道；卷积运算以connectModel为准
	bool* connectModel; // [输入通道*outChannels+输出通道]，true为有效连接

	// v、y、d形状均为[输出通道][输出行][输出列]。
	float*** v; // 加偏置后的z，也是激活函数的输入值
	float*** y; // 激活函数后神经元的输出

	// 每个输出位置一个dL/dz，与卷积核权重梯度不同。
	float*** d; // dL/dz，z为激活前且含偏置的值
}CovLayer;

// 可训练下采样：窗口求和、乘a、加b、再激活；不是最大池化。
typedef struct pooling_layer{
	int inputWidth;   // 输入特征图宽度
	int inputHeight;  // 输入特征图高度
	int mapSize;      // 正方形窗口边长，也作为采样步长

	int inChannels;   // 输入通道数，不是样本数
	int outChannels;  // 输出通道数，不是样本数

	float* scaleData;   // 每通道的可训练缩放系数a
	float* basicData;   // 每通道共享的偏置b
	float* dscaleData;  // 损失对a的梯度
	float* dbasicData;  // 损失对b的梯度

	// 三维数组形状：[outChannels][inputHeight/mapSize][inputWidth/mapSize]。
	float*** sumData; // 每个窗口的输入之和，用于计算缩放系数梯度
	float*** y; // y=f(a*sum+b)，f为缩放tanh
	float*** d; // 反传先累加dL/dy，再转换为dL/dz，z=a*sum+b
}PoolLayer;

// 全连接层：F6使用
typedef struct nn_layer{
	int inputNum;   // 输入向量长度
	int outputNum;  // 输出神经元数

	float** wData; // wData[输出神经元][输入分量]，形状[outputNum][inputNum]
	float* basicData;   // 每个输出神经元一个偏置

	// v、y、d均为长度outputNum的一维数组。
	float* v; // 加偏置后的z，也是激活函数的输入值
	float* y; // 激活函数后神经元的输出
	float* d; // dL/dz，z为激活前且含偏置的值

	bool isFullConnect; // 当前恒为true，全连接层没有稀疏掩码
}OutLayer;

typedef struct cnn_network{
	int layerNum; // 7个计算层，不含输入
	CovLayer* C1; // 32×32×1 -> 28×28×6
	PoolLayer* S2; // 28×28×6 -> 14×14×6
	CovLayer* C3; // 14×14×6 -> 10×10×16，60组有效连接
	PoolLayer* S4; // 10×10×16 -> 5×5×16
	CovLayer* C5; // 16个5×5输入通道 -> 120个1×1输出
	OutLayer* F6; // C5的120维输出 -> 84维缩放tanh
	RbfLayer O7; // 固定84维模板 -> 10个平方距离

	float* L; // 仅cnntrain分配；平滑MAP损失长度trainNum，每轮覆盖，不用于求导
}CNN;

typedef struct train_opts{
	int numepochs; // epoch数，每轮遍历trainNum个样本
	float alpha; // 普通SGD学习率
}CNNOpts;

// 调用方先令CNN对象为零，并设置一次srand(seed)；当前结构使用{32,32}、10类。
void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);
/* 训练前trainNum个样本，每个样本更新一次；不在此函数中保存模型。
 * 图片与标签须一一对应，trainNum须为正且不超过数组长度，调用前须清空中间量。 */
void cnntrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum,FILE* log); // log可为NULL；否则同步记录进度和耗时
// 评估前testNum个样本，返回错误率；数量须有效，首次前向前须清空中间量。
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum);
// 保存结构、RBF配置及参数，不保存特征图或训练进度。
void savecnn(CNN* cnn, const char* filename);
// 向已经cnnsetup分配好的网络载入兼容模型。
void importcnn(CNN* cnn, const char* filename);

// 分配并初始化通道全连接的卷积层，步长1、无填充。
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels);
// 复制连接表并清零断开槽；不重新初始化新接通的权重。
void CovLayerConnect(CovLayer* covL,const bool* connectModel);
bool covConnected(const CovLayer* layer,int input,int output);
// 分配下采样层；当前要求输入输出通道相等、尺寸可被窗口整除。
PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels);
// 历史遗留声明：当前无实现且没有调用，下采样不使用通道连接表。
void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
// 分配全连接层；OutLayer是历史命名，当前用于F6。
OutLayer* initOutLayer(int inputNum,int outputNum);

// 只负责激活；z由各层先完成加偏置。
float activation_Sigma(float z); // LeNet-5缩放tanh：1.7159*tanh((2/3)*z)

void poolff(PoolLayer* pool,float*** input); // 可训练下采样前向计算
void poolbp(PoolLayer* pool); // 将dL/dy转换为dL/dz，并求a、b的梯度
void poolapplygrads(PoolLayer* pool,float alpha);

// inputData[行][列]为32×32预处理图片；后续样本调用前须先cnnclear。
void cnnff(CNN* cnn,float** inputData); // 网络的前向传播
double cnnloss(const CNN* cnn,const float* outputData); // 根据10维one-hot标签读取当前前向结果的MAP损失
void cnnbp(CNN* cnn,float* outputData); // 计算各层d及下采样参数梯度，不更新参数
// 完整反传后，结合缓存求卷积/全连接参数梯度并执行SGD；inputData须为同一样本。
void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData);
void cnnclear(CNN* cnn); // 清空中间结果和局部梯度，不清权重、偏置和损失曲线

// nnSize.c是输入长度，nnSize.r是输出长度；计算Wx+b，不执行激活。
void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize); // 单层全连接神经网络的前向传播

void savecnndata(CNN* cnn,const char* filename,float** inputdata); // 历史调试转储，与savecnn格式不同，不能用importcnn加载

#endif
