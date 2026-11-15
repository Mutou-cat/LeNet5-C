#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "cnn.h"

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize)
{
	cnn->layerNum=5;

	nSize inSize;
	int mapSize=5;
	inSize.c=inputSize.c;
	inSize.r=inputSize.r;
	cnn->C1=initCovLayer(inSize.c,inSize.r,5,1,6);
	inSize.c=inSize.c-mapSize+1;
	inSize.r=inSize.r-mapSize+1;
	cnn->S2=initPoolLayer(inSize.c,inSize.r,2,6,6,AvePool);
	inSize.c=inSize.c/2;
	inSize.r=inSize.r/2;
	cnn->C3=initCovLayer(inSize.c,inSize.r,5,6,12);
	inSize.c=inSize.c-mapSize+1;
	inSize.r=inSize.r-mapSize+1;
	cnn->S4=initPoolLayer(inSize.c,inSize.r,2,12,12,AvePool);
	inSize.c=inSize.c/2;
	inSize.r=inSize.r/2;
	cnn->O5=initOutLayer(inSize.c*inSize.r*12,outputSize);

	cnn->e=(float*)calloc(cnn->O5->outputNum,sizeof(float));
}

CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels)
{
	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));

	covL->inputHeight=inputHeight;
	covL->inputWidth=inputWidth;
	covL->mapSize=mapSize;

	covL->inChannels=inChannels;
	covL->outChannels=outChannels;

	covL->isFullConnect=true; // 默认为全连接

	// 权重空间的初始化，先行再列调用，[r][c]
	int i,j,c,r;
	srand((unsigned)time(NULL));
	covL->mapData=(float****)malloc(inChannels*sizeof(float***));
	for(i=0;i<inChannels;i++){
		covL->mapData[i]=(float***)malloc(outChannels*sizeof(float**));
		for(j=0;j<outChannels;j++){
			covL->mapData[i][j]=(float**)malloc(mapSize*sizeof(float*));
			for(r=0;r<mapSize;r++){
				covL->mapData[i][j][r]=(float*)malloc(mapSize*sizeof(float));
				for(c=0;c<mapSize;c++){
					float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2;
					covL->mapData[i][j][r][c]=randnum*sqrt((float)6.0/(float)(mapSize*mapSize*(inChannels+outChannels)));
				}
			}
		}
	}
	// 权重梯度变化
	covL->dmapData=(float****)malloc(inChannels*sizeof(float***));
	for(i=0;i<inChannels;i++){
		covL->dmapData[i]=(float***)malloc(outChannels*sizeof(float**));
		for(j=0;j<outChannels;j++){
			covL->dmapData[i][j]=(float**)malloc(mapSize*sizeof(float*));
			for(r=0;r<mapSize;r++){
				covL->dmapData[i][j][r]=(float*)calloc(mapSize,sizeof(float));
			}
		}
	}

	covL->basicData=(float*)calloc(outChannels,sizeof(float));

	int outW=inputWidth-mapSize+1;
	int outH=inputHeight-mapSize+1;


	covL->d=(float***)malloc(outChannels*sizeof(float**));
	covL->v=(float***)malloc(outChannels*sizeof(float**));
	covL->y=(float***)malloc(outChannels*sizeof(float**));
	for(j=0;j<outChannels;j++){
		covL->d[j]=(float**)malloc(outH*sizeof(float*));
		covL->v[j]=(float**)malloc(outH*sizeof(float*));
		covL->y[j]=(float**)malloc(outH*sizeof(float*));
		for(r=0;r<outH;r++){
			covL->d[j][r]=(float*)calloc(outW,sizeof(float));
			covL->v[j][r]=(float*)calloc(outW,sizeof(float));
			covL->y[j][r]=(float*)calloc(outW,sizeof(float));
		}
	}

	return covL;
}

PoolLayer* initPoolLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int poolType)
{
	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));

	poolL->inputHeight=inputHeight;
	poolL->inputWidth=inputWidth;
	poolL->mapSize=mapSize;
	poolL->inChannels=inChannels;
	poolL->outChannels=outChannels;
	poolL->poolType=poolType;

	poolL->scaleData=(float*)malloc(outChannels*sizeof(float));
	poolL->basicData=(float*)calloc(outChannels,sizeof(float));
	poolL->dscaleData=(float*)calloc(outChannels,sizeof(float));
	poolL->dbasicData=(float*)calloc(outChannels,sizeof(float));
	for(int channel=0;channel<outChannels;channel++)
		poolL->scaleData[channel]=1.0f/(mapSize*mapSize);

	int outW=inputWidth/mapSize;
	int outH=inputHeight/mapSize;

	int j,r;
	poolL->sumData=(float***)malloc(outChannels*sizeof(float**));
	poolL->d=(float***)malloc(outChannels*sizeof(float**));
	poolL->y=(float***)malloc(outChannels*sizeof(float**));
	for(j=0;j<outChannels;j++){
		poolL->sumData[j]=(float**)malloc(outH*sizeof(float*));
		poolL->d[j]=(float**)malloc(outH*sizeof(float*));
		poolL->y[j]=(float**)malloc(outH*sizeof(float*));
		for(r=0;r<outH;r++){
			poolL->sumData[j][r]=(float*)calloc(outW,sizeof(float));
			poolL->d[j][r]=(float*)calloc(outW,sizeof(float));
			poolL->y[j][r]=(float*)calloc(outW,sizeof(float));
		}
	}

	return poolL;
}

OutLayer* initOutLayer(int inputNum,int outputNum)
{
	OutLayer* outL=(OutLayer*)malloc(sizeof(OutLayer));

	outL->inputNum=inputNum;
	outL->outputNum=outputNum;


	outL->basicData=(float*)calloc(outputNum,sizeof(float));

	outL->d=(float*)calloc(outputNum,sizeof(float));
	outL->v=(float*)calloc(outputNum,sizeof(float));
	outL->y=(float*)calloc(outputNum,sizeof(float));

	// 权重的初始化
	outL->wData=(float**)malloc(outputNum*sizeof(float*)); // 输入行，输出列
	int i,j;
	srand((unsigned)time(NULL));
	for(i=0;i<outputNum;i++){
		outL->wData[i]=(float*)malloc(inputNum*sizeof(float));
		for(j=0;j<inputNum;j++){
			float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; // 产生一个-1到1的随机数
			outL->wData[i][j]=randnum*sqrt((float)6.0/(float)(inputNum+outputNum));
		}
	}

	outL->isFullConnect=true;

	return outL;
}

int vecmaxIndex(float* vec, int veclength)// 返回向量最大数的序号
{
	int i;
	float maxnum=-1.0;
	int maxIndex=0;
	for(i=0;i<veclength;i++){
		if(maxnum<vec[i]){
			maxnum=vec[i];
			maxIndex=i;
		}
	}
	return maxIndex;
}

// 测试cnn函数
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum)
{
	int n=0;
	int incorrectnum=0;  //错误预测的数目
	for(n=0;n<testNum;n++){
		cnnff(cnn,inputData->ImgPtr[n].ImgData);
		if(vecmaxIndex(cnn->O5->y,cnn->O5->outputNum)!=vecmaxIndex(outputData->LabelPtr[n].LabelData,cnn->O5->outputNum))
			incorrectnum++;
		cnnclear(cnn);
	}
	return (float)incorrectnum/(float)testNum;
}

// 保存cnn
void savecnn(CNN* cnn, const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}

	int i,j,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				fwrite(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	fwrite(cnn->C1->basicData,sizeof(float),cnn->C1->outChannels,fp);

	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				fwrite(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	fwrite(cnn->C3->basicData,sizeof(float),cnn->C3->outChannels,fp);

	// O5输出层
	for(i=0;i<cnn->O5->outputNum;i++)
		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
	fwrite(cnn->O5->basicData,sizeof(float),cnn->O5->outputNum,fp);

	fclose(fp);
}
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}

	int i,j,c,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				for(c=0;c<cnn->C1->mapSize;c++){
					read_exact(&cnn->C1->mapData[i][j][r][c],sizeof(float),1,fp);
				}

	for(i=0;i<cnn->C1->outChannels;i++)
		read_exact(&cnn->C1->basicData[i],sizeof(float),1,fp);

	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				for(c=0;c<cnn->C3->mapSize;c++)
				read_exact(&cnn->C3->mapData[i][j][r][c],sizeof(float),1,fp);

	for(i=0;i<cnn->C3->outChannels;i++)
		read_exact(&cnn->C3->basicData[i],sizeof(float),1,fp);

	// O5输出层
	for(i=0;i<cnn->O5->outputNum;i++)
		for(j=0;j<cnn->O5->inputNum;j++)
			read_exact(&cnn->O5->wData[i][j],sizeof(float),1,fp);

	for(i=0;i<cnn->O5->outputNum;i++)
		read_exact(&cnn->O5->basicData[i],sizeof(float),1,fp);

	fclose(fp);
}

void cnntrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum)
{
	// 学习训练误差曲线
	cnn->L=(float*)malloc(trainNum*sizeof(float));
	printf("Training: %d samples, %d epochs, learning rate %.4g\n",
		trainNum, opts.numepochs, opts.alpha);
	fflush(stdout);
	int e;
	for(e=0;e<opts.numepochs;e++){
		printf("Epoch %d/%d: starting\n", e+1, opts.numepochs);
		fflush(stdout);
		int n=0;
		for(n=0;n<trainNum;n++){
			cnnff(cnn,inputData->ImgPtr[n].ImgData);  // 前向传播，这里主要计算各
			cnnbp(cnn,outputData->LabelPtr[n].LabelData); // 后向传播，这里主要计算各神经元的误差梯度


			cnnapplygrads(cnn,opts,inputData->ImgPtr[n].ImgData); // 更新权重

			cnnclear(cnn);
			// 计算并保存误差能量
			float l=0.0;
			int i;
			for(i=0;i<cnn->O5->outputNum;i++)
				l=l+cnn->e[i]*cnn->e[i];
			if(n==0)
				cnn->L[n]=l/(float)2.0;
			else
				cnn->L[n]=cnn->L[n-1]*0.99+0.01*l/(float)2.0;
			// 首个样本、每 100 个样本及本轮结束时输出平滑损失。
			if(n==0 || (n+1)%100==0 || n+1==trainNum){
				printf("Epoch %d/%d | %d/%d (%6.2f%%) | smoothed loss: %.6f\n",
					e+1, opts.numepochs, n+1, trainNum,
					100.0*(n+1)/trainNum, cnn->L[n]);
				fflush(stdout);
			}
		}
	}
}

// 这里InputData是图像数据，inputData[r][c],r行c列，这里根各权重模板是一致的
void cnnff(CNN* cnn,float** inputData)
{
	// 第一层的传播
	int i,j,r,c;
	// 第一层输出数据
	nSize mapSize={cnn->C1->mapSize,cnn->C1->mapSize};
	nSize inSize={cnn->C1->inputWidth,cnn->C1->inputHeight};
	nSize outSize={cnn->S2->inputWidth,cnn->S2->inputHeight};
	for(i=0;i<(cnn->C1->outChannels);i++){
		for(j=0;j<(cnn->C1->inChannels);j++){
			float** mapout=cov(cnn->C1->mapData[j][i],mapSize,inputData,inSize,valid);
			addmat(cnn->C1->v[i],cnn->C1->v[i],outSize,mapout,outSize);
			for(r=0;r<outSize.r;r++)
				free(mapout[r]);
			free(mapout);
		}
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				cnn->C1->y[i][r][c]=activation_Sigma(cnn->C1->v[i][r][c],cnn->C1->basicData[i]);
	}

	// S2：窗口求和、乘每通道系数、加偏置、缩放tanh。
	poolff(cnn->S2,cnn->C1->y);

	// 第三层输出传播,这里是全连接
	outSize.c=cnn->S4->inputWidth;
	outSize.r=cnn->S4->inputHeight;
	inSize.c=cnn->C3->inputWidth;
	inSize.r=cnn->C3->inputHeight;
	mapSize.c=cnn->C3->mapSize;
	mapSize.r=cnn->C3->mapSize;
	for(i=0;i<(cnn->C3->outChannels);i++){
		for(j=0;j<(cnn->C3->inChannels);j++){
			float** mapout=cov(cnn->C3->mapData[j][i],mapSize,cnn->S2->y[j],inSize,valid);
			addmat(cnn->C3->v[i],cnn->C3->v[i],outSize,mapout,outSize);
			for(r=0;r<outSize.r;r++)
				free(mapout[r]);
			free(mapout);
		}
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				cnn->C3->y[i][r][c]=activation_Sigma(cnn->C3->v[i][r][c],cnn->C3->basicData[i]);
	}

	// S4采用与S2相同的可训练下采样。
	poolff(cnn->S4,cnn->C3->y);
	outSize.c=cnn->S4->inputWidth/cnn->S4->mapSize;
	outSize.r=cnn->S4->inputHeight/cnn->S4->mapSize;

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float* O5inData=(float*)malloc((cnn->O5->inputNum)*sizeof(float));
	for(i=0;i<(cnn->S4->outChannels);i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				O5inData[i*outSize.r*outSize.c+r*outSize.c+c]=cnn->S4->y[i][r][c];

	nSize nnSize={cnn->O5->inputNum,cnn->O5->outputNum};
	nnff(cnn->O5->v,O5inData,cnn->O5->wData,cnn->O5->basicData,nnSize);
	for(i=0;i<cnn->O5->outputNum;i++)
		cnn->O5->y[i]=activation_Sigma(cnn->O5->v[i],cnn->O5->basicData[i]);
	free(O5inData);
}

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bas) // LeNet-5缩放tanh激活函数
{
	const float A=1.7159f;
	const float S=2.0f/3.0f;
	float z=input+bas;
	return A*tanhf(S*z);
}

// LeNet-5下采样：每通道共享a、b；缓存窗口和供反向传播使用。
void poolff(PoolLayer* pool,float*** input)
{
	int width=pool->inputWidth/pool->mapSize;
	int height=pool->inputHeight/pool->mapSize;
	for(int i=0;i<pool->outChannels;i++)
		for(int r=0;r<height;r++)
			for(int c=0;c<width;c++) {
				float sum=0.0f;
				for(int dr=0;dr<pool->mapSize;dr++)
					for(int dc=0;dc<pool->mapSize;dc++)
						sum+=input[i][r*pool->mapSize+dr][c*pool->mapSize+dc];
				pool->sumData[i][r][c]=sum;
				pool->y[i][r][c]=activation_Sigma(pool->scaleData[i]*sum,pool->basicData[i]);
			}
}

void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize) // 求平均值
{
	int outputW=inputSize.c/mapSize;
	int outputH=inputSize.r/mapSize;
	if(outputSize.c!=outputW||outputSize.r!=outputH)
		printf("ERROR: output size is wrong!!");

	int i,j,m,n;
	for(i=0;i<outputH;i++)
		for(j=0;j<outputW;j++)
		{
			float sum=0.0;
			for(m=i*mapSize;m<i*mapSize+mapSize;m++)
				for(n=j*mapSize;n<j*mapSize+mapSize;n++)
					sum=sum+input[m][n];

			output[i][j]=sum/(float)(mapSize*mapSize);
		}
}

// 单层全连接神经网络的前向传播
float vecMulti(float* vec1,float* vec2,int vecL)// 两向量相乘
{
	int i;
	float m=0;
	for(i=0;i<vecL;i++)
		m=m+vec1[i]*vec2[i];
	return m;
}

void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize)
{
	int w=nnSize.c;
	int h=nnSize.r;

	int i;
	for(i=0;i<h;i++)
		output[i]=vecMulti(input,wdata[i],w)+bas[i];
}

float sigma_derivation(float y){ // 缩放tanh对激活前输入z的导数
	const float A=1.7159f;
	const float S=2.0f/3.0f;
	float t=y/A; // 参数y是激活后的输出：y=A*tanh(S*z)
	return A*S*(1.0f-t*t);
}

// 调用前d是dL/dy，调用后d是dL/dz；每个样本重新计算参数梯度。
void poolbp(PoolLayer* pool)
{
	int width=pool->inputWidth/pool->mapSize;
	int height=pool->inputHeight/pool->mapSize;
	for(int i=0;i<pool->outChannels;i++) {
		pool->dscaleData[i]=0.0f;
		pool->dbasicData[i]=0.0f;
		for(int r=0;r<height;r++)
			for(int c=0;c<width;c++) {
				pool->d[i][r][c]*=sigma_derivation(pool->y[i][r][c]);
				pool->dscaleData[i]+=pool->d[i][r][c]*pool->sumData[i][r][c];
				pool->dbasicData[i]+=pool->d[i][r][c];
			}
	}
}

void poolapplygrads(PoolLayer* pool,float alpha)
{
	for(int i=0;i<pool->outChannels;i++) {
		pool->scaleData[i]-=alpha*pool->dscaleData[i];
		pool->basicData[i]-=alpha*pool->dbasicData[i];
	}
}

void cnnbp(CNN* cnn,float* outputData) // 网络的后向传播
{
	int i,j,c,r; // 将误差保存到网络中
	for(i=0;i<cnn->O5->outputNum;i++)
		cnn->e[i]=cnn->O5->y[i]-outputData[i];

	/*从后向前反向计算*/
	// 输出层O5
	for(i=0;i<cnn->O5->outputNum;i++)
		cnn->O5->d[i]=cnn->e[i]*sigma_derivation(cnn->O5->y[i]);

	// S4层，传递到S4层的误差
	// 先累加损失对S4输出y的梯度，再通过poolbp转换为对z的梯度。
	nSize outSize={cnn->S4->inputWidth/cnn->S4->mapSize,cnn->S4->inputHeight/cnn->S4->mapSize};
	for(i=0;i<cnn->S4->outChannels;i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				for(j=0;j<cnn->O5->outputNum;j++){
					int wInt=i*outSize.c*outSize.r+r*outSize.c+c;
					cnn->S4->d[i][r][c]=cnn->S4->d[i][r][c]+cnn->O5->d[j]*cnn->O5->wData[j][wInt];
				}

	poolbp(cnn->S4);

	// C3层
	// 由S4层传递的各反向误差,这里只是在S4的梯度上扩充一倍
	nSize S4dSize={cnn->S4->inputWidth/cnn->S4->mapSize,cnn->S4->inputHeight/cnn->S4->mapSize};
	// z=a*sum+b，反向到每个输入的系数是a，不再固定除以窗口面积。
	for(i=0;i<cnn->C3->outChannels;i++){
		float** C3e=UpSample(cnn->S4->d[i],S4dSize,cnn->S4->mapSize,cnn->S4->mapSize);
		for(r=0;r<cnn->S4->inputHeight;r++)
			for(c=0;c<cnn->S4->inputWidth;c++)
				cnn->C3->d[i][r][c]=C3e[r][c]*cnn->S4->scaleData[i]*sigma_derivation(cnn->C3->y[i][r][c]);
		for(r=0;r<cnn->S4->inputHeight;r++)
			free(C3e[r]);
		free(C3e);
	}

	// S2层：先累加dL/dy，再反传其缩放tanh。
	// 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
	outSize.c=cnn->C3->inputWidth;
	outSize.r=cnn->C3->inputHeight;
	nSize inSize={cnn->S4->inputWidth,cnn->S4->inputHeight};
	nSize mapSize={cnn->C3->mapSize,cnn->C3->mapSize};
	for(i=0;i<cnn->S2->outChannels;i++){
		for(j=0;j<cnn->C3->outChannels;j++){
			float** corr=correlation(cnn->C3->mapData[i][j],mapSize,cnn->C3->d[j],inSize,full);
			addmat(cnn->S2->d[i],cnn->S2->d[i],outSize,corr,outSize);
			for(r=0;r<outSize.r;r++)
				free(corr[r]);
			free(corr);
		}
	}
	poolbp(cnn->S2);

	// C1层，卷积层
	nSize S2dSize={cnn->S2->inputWidth/cnn->S2->mapSize,cnn->S2->inputHeight/cnn->S2->mapSize};
	// z=a*sum+b，反向到每个输入的系数是a，不再固定除以窗口面积。
	for(i=0;i<cnn->C1->outChannels;i++){
		float** C1e=UpSample(cnn->S2->d[i],S2dSize,cnn->S2->mapSize,cnn->S2->mapSize);
		for(r=0;r<cnn->S2->inputHeight;r++)
			for(c=0;c<cnn->S2->inputWidth;c++)
				cnn->C1->d[i][r][c]=C1e[r][c]*cnn->S2->scaleData[i]*sigma_derivation(cnn->C1->y[i][r][c]);
		for(r=0;r<cnn->S2->inputHeight;r++)
			free(C1e[r]);
		free(C1e);
	}
}

void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData) // 更新权重
{
	// 全部反传完成后，统一更新卷积、输出层和下采样层参数。
	int i,j,r,c;

	// C1层的权重更新
	nSize dSize={cnn->S2->inputHeight,cnn->S2->inputWidth};
	nSize ySize={cnn->C1->inputHeight,cnn->C1->inputWidth};
	nSize mapSize={cnn->C1->mapSize,cnn->C1->mapSize};

	for(i=0;i<cnn->C1->outChannels;i++){
		for(j=0;j<cnn->C1->inChannels;j++){
			float** flipinputData=rotate180(inputData,ySize);
			float** C1dk=cov(cnn->C1->d[i],dSize,flipinputData,ySize,valid);
			multifactor(C1dk,C1dk,mapSize,-1*opts.alpha);
			addmat(cnn->C1->mapData[j][i],cnn->C1->mapData[j][i],mapSize,C1dk,mapSize);
			for(r=0;r<mapSize.r;r++)
				free(C1dk[r]);
			free(C1dk);
			for(r=0;r<ySize.r;r++)
				free(flipinputData[r]);
			free(flipinputData);
		}
		cnn->C1->basicData[i]=cnn->C1->basicData[i]-opts.alpha*summat(cnn->C1->d[i],dSize);
	}

	// C3层的权重更新
	dSize.c=cnn->S4->inputWidth;
	dSize.r=cnn->S4->inputHeight;
	ySize.c=cnn->C3->inputWidth;
	ySize.r=cnn->C3->inputHeight;
	mapSize.c=cnn->C3->mapSize;
	mapSize.r=cnn->C3->mapSize;
	for(i=0;i<cnn->C3->outChannels;i++){
		for(j=0;j<cnn->C3->inChannels;j++){
			float** flipinputData=rotate180(cnn->S2->y[j],ySize);
			float** C3dk=cov(cnn->C3->d[i],dSize,flipinputData,ySize,valid);
			multifactor(C3dk,C3dk,mapSize,-1.0*opts.alpha);
			addmat(cnn->C3->mapData[j][i],cnn->C3->mapData[j][i],mapSize,C3dk,mapSize);
			for(r=0;r<mapSize.r;r++)
				free(C3dk[r]);
			free(C3dk);
			for(r=0;r<ySize.r;r++)
				free(flipinputData[r]);
			free(flipinputData);
		}
		cnn->C3->basicData[i]=cnn->C3->basicData[i]-opts.alpha*summat(cnn->C3->d[i],dSize);
	}

	// 输出层
	// 首先需要将前面的多维输出展开成一维向量
	float* O5inData=(float*)malloc((cnn->O5->inputNum)*sizeof(float));
	nSize outSize={cnn->S4->inputWidth/cnn->S4->mapSize,cnn->S4->inputHeight/cnn->S4->mapSize};
	for(i=0;i<(cnn->S4->outChannels);i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				O5inData[i*outSize.r*outSize.c+r*outSize.c+c]=cnn->S4->y[i][r][c];

	for(j=0;j<cnn->O5->outputNum;j++){
		for(i=0;i<cnn->O5->inputNum;i++)
			cnn->O5->wData[j][i]=cnn->O5->wData[j][i]-opts.alpha*cnn->O5->d[j]*O5inData[i];
		cnn->O5->basicData[j]=cnn->O5->basicData[j]-opts.alpha*cnn->O5->d[j];
	}
	poolapplygrads(cnn->S2,opts.alpha);
	poolapplygrads(cnn->S4,opts.alpha);
	free(O5inData);
}

void cnnclear(CNN* cnn)
{
	// 将神经元的部分数据清除
	int j,c,r;
	// C1网络
	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			for(c=0;c<cnn->S2->inputWidth;c++){
				cnn->C1->d[j][r][c]=(float)0.0;
				cnn->C1->v[j][r][c]=(float)0.0;
				cnn->C1->y[j][r][c]=(float)0.0;
			}
		}
	}
	// S2网络
	for(j=0;j<cnn->S2->outChannels;j++){
		cnn->S2->dscaleData[j]=0.0f;
		cnn->S2->dbasicData[j]=0.0f;
		for(r=0;r<cnn->C3->inputHeight;r++){
			for(c=0;c<cnn->C3->inputWidth;c++){
				cnn->S2->sumData[j][r][c]=0.0f;
				cnn->S2->d[j][r][c]=(float)0.0;
				cnn->S2->y[j][r][c]=(float)0.0;
			}
		}
	}
	// C3网络
	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			for(c=0;c<cnn->S4->inputWidth;c++){
				cnn->C3->d[j][r][c]=(float)0.0;
				cnn->C3->v[j][r][c]=(float)0.0;
				cnn->C3->y[j][r][c]=(float)0.0;
			}
		}
	}
	// S4网络
	for(j=0;j<cnn->S4->outChannels;j++){
		cnn->S4->dscaleData[j]=0.0f;
		cnn->S4->dbasicData[j]=0.0f;
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
				cnn->S4->sumData[j][r][c]=0.0f;
				cnn->S4->d[j][r][c]=(float)0.0;
				cnn->S4->y[j][r][c]=(float)0.0;
			}
		}
	}
	// O5输出
	for(j=0;j<cnn->O5->outputNum;j++){
		cnn->O5->d[j]=(float)0.0;
		cnn->O5->v[j]=(float)0.0;
		cnn->O5->y[j]=(float)0.0;
	}
}

// 这是用于测试的函数
void savecnndata(CNN* cnn,const char* filename,float** inputdata) // 保存CNN网络中的相关数据
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}

	// C1的数据
	int i,j,r;
	// C1网络
	for(i=0;i<cnn->C1->inputHeight;i++)
		fwrite(inputdata[i],sizeof(float),cnn->C1->inputWidth,fp);
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				fwrite(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	fwrite(cnn->C1->basicData,sizeof(float),cnn->C1->outChannels,fp);

	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			fwrite(cnn->C1->v[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
		for(r=0;r<cnn->S2->inputHeight;r++){
			fwrite(cnn->C1->d[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
		for(r=0;r<cnn->S2->inputHeight;r++){
			fwrite(cnn->C1->y[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
	}

	// S2网络
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->d[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->y[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
	}
	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				fwrite(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	fwrite(cnn->C3->basicData,sizeof(float),cnn->C3->outChannels,fp);

	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			fwrite(cnn->C3->v[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
		for(r=0;r<cnn->S4->inputHeight;r++){
			fwrite(cnn->C3->d[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
		for(r=0;r<cnn->S4->inputHeight;r++){
			fwrite(cnn->C3->y[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
	}

	// S4网络
	for(j=0;j<cnn->S4->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			fwrite(cnn->S4->d[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
		}
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			fwrite(cnn->S4->y[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
		}
	}

	// O5输出层
	for(i=0;i<cnn->O5->outputNum;i++)
		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
	fwrite(cnn->O5->basicData,sizeof(float),cnn->O5->outputNum,fp);
	fwrite(cnn->O5->v,sizeof(float),cnn->O5->outputNum,fp);
	fwrite(cnn->O5->d,sizeof(float),cnn->O5->outputNum,fp);
	fwrite(cnn->O5->y,sizeof(float),cnn->O5->outputNum,fp);

	fclose(fp);
}
