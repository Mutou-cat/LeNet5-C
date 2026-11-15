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


// 卷积层
typedef struct convolutional_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小，模板一般都是正方形

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	// 关于特征模板的权重分布，这里是一个四维数组
	// 其大小为inChannels*outChannels*mapSize*mapSize大小
	// 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
	// 这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
	float**** mapData;     //存放特征模块的数据
	float**** dmapData;    //存放特征模块的数据的局部梯度

	float* basicData;   //偏置，偏置的大小，为outChannels
	bool isFullConnect; //是否为全连接
	bool* connectModel; //连接模式（默认为全连接）

	// 下面三者的大小同输出的维度相同
	float*** v; // 进入激活函数的输入值
	float*** y; // 激活函数后神经元的输出

	// 输出像素的局部梯度
	float*** d; // 网络的局部梯度,δ值
}CovLayer;

// 采样层 pooling
typedef struct pooling_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	float* scaleData;   // 每通道的可训练缩放系数a
	float* basicData;   // 每通道共享的偏置b
	float* dscaleData;  // 损失对a的梯度
	float* dbasicData;  // 损失对b的梯度

	float*** sumData; // 每个窗口的输入之和，用于计算缩放系数梯度
	float*** y; // y=f(a*sum+b)，f为缩放tanh
	float*** d; // 反传先累加dL/dy，再转换为dL/dz，z=a*sum+b
}PoolLayer;

// 输出层 全连接的神经网络
typedef struct nn_layer{
	int inputNum;   //输入数据的数目
	int outputNum;  //输出数据的数目

	float** wData; // 权重数据，为一个inputNum*outputNum大小
	float* basicData;   //偏置，大小为outputNum大小

	// 下面三者的大小同输出的维度相同
	float* v; // 进入激活函数的输入值
	float* y; // 激活函数后神经元的输出
	float* d; // 网络的局部梯度,δ值

	bool isFullConnect; //是否为全连接
}OutLayer;

typedef struct cnn_network{
	int layerNum;
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;

	float* e; // 训练误差
	float* L; // 瞬时误差能量
}CNN;

typedef struct train_opts{
	int numepochs; // 训练的迭代次数
	float alpha; // 学习速率
}CNNOpts;

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);
/*
	CNN网络的训练函数
	inputData，outputData分别存入训练数据
	dataNum表明数据数目
*/
void cnntrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum);
// 测试cnn函数
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum);
// 保存cnn
void savecnn(CNN* cnn, const char* filename);
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename);

// 初始化卷积层
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels);
void CovLayerConnect(CovLayer* covL,bool* connectModel);
// 初始化采样层
PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels);
void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
// 初始化输出层
OutLayer* initOutLayer(int inputNum,int outputNum);

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bas); // LeNet-5缩放tanh：1.7159*tanh((2/3)*(input+bas))

void poolff(PoolLayer* pool,float*** input); // 可训练下采样前向计算
void poolbp(PoolLayer* pool); // 将dL/dy转换为dL/dz，并求a、b的梯度
void poolapplygrads(PoolLayer* pool,float alpha);

void cnnff(CNN* cnn,float** inputData); // 网络的前向传播
void cnnbp(CNN* cnn,float* outputData); // 网络的后向传播
void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData);
void cnnclear(CNN* cnn); // 将数据vyd清零

/*
	单层全连接神经网络的处理
	nnSize是网络的大小
*/
void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize); // 单层全连接神经网络的前向传播

void savecnndata(CNN* cnn,const char* filename,float** inputdata); // 保存CNN网络中的相关数据

#endif
