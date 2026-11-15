#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "cnn.h"

// LeNet-5论文Table I：每行对应一个C3输出通道，列出其S2输入通道。
// 位k=1表示连接输入通道k；输出0..5连3路，6..14连4路，15连6路。
static const unsigned char c3_connections[16]={
	0x07,0x0e,0x1c,0x38,0x31,0x23,
	0x0f,0x1e,0x3c,0x39,0x33,0x27,
	0x1b,0x36,0x2d,0x3f
};

bool covConnected(const CovLayer* layer,int input,int output)
{
	return layer->connectModel[input*layer->outChannels+output];
}

// 拷贝连接表；保留四维数组寻址，断开的权重槽清零且不参与运算。
void CovLayerConnect(CovLayer* layer,const bool* connections)
{
	layer->isFullConnect=true;
	for(int i=0;i<layer->inChannels;i++)
		for(int j=0;j<layer->outChannels;j++) {
			bool connected=connections[i*layer->outChannels+j];
			layer->connectModel[i*layer->outChannels+j]=connected;
			if(!connected) {
				layer->isFullConnect=false;
				for(int r=0;r<layer->mapSize;r++)
					for(int c=0;c<layer->mapSize;c++) {
						layer->mapData[i][j][r][c]=0.0f;
						layer->dmapData[i][j][r][c]=0.0f;
					}
			}
		}
}

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize)
{
	if(outputSize!=RBF_CLASSES) {
		fprintf(stderr,"RBF output requires 10 digit classes\n");exit(EXIT_FAILURE);
	}
	cnn->layerNum=7;

	nSize inSize;
	int mapSize=5;
	inSize.c=inputSize.c;
	inSize.r=inputSize.r;
	cnn->C1=initCovLayer(inSize.c,inSize.r,5,1,6);
	inSize.c=inSize.c-mapSize+1;
	inSize.r=inSize.r-mapSize+1;
	cnn->S2=initPoolLayer(inSize.c,inSize.r,2,6,6);
	inSize.c=inSize.c/2;
	inSize.r=inSize.r/2;
	cnn->C3=initCovLayer(inSize.c,inSize.r,5,6,16);
	bool connections[6*16];
	for(int i=0;i<6;i++)
		for(int j=0;j<16;j++)
			connections[i*16+j]=(c3_connections[j] & (1u<<i)) != 0;
	CovLayerConnect(cnn->C3,connections);
	inSize.c=inSize.c-mapSize+1;
	inSize.r=inSize.r-mapSize+1;
	cnn->S4=initPoolLayer(inSize.c,inSize.r,2,16,16);
	inSize.c=inSize.c/2;
	inSize.r=inSize.r/2;
	cnn->C5=initCovLayer(inSize.c,inSize.r,5,cnn->S4->outChannels,120);
	inSize.c=inSize.c-cnn->C5->mapSize+1;
	inSize.r=inSize.r-cnn->C5->mapSize+1;
	cnn->F6=initOutLayer(inSize.c*inSize.r*cnn->C5->outChannels,84);
	memset(&cnn->O7,0,sizeof(cnn->O7));
}

CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels)
{
	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));

	covL->inputHeight=inputHeight;
	covL->inputWidth=inputWidth;
	covL->mapSize=mapSize;

	covL->inChannels=inChannels;
	covL->outChannels=outChannels;

	covL->isFullConnect=true; // 默认所有输入/输出通道连接，C3随后设置部分连接
	covL->connectModel=(bool*)malloc(inChannels*outChannels*sizeof(bool));
	for(int k=0;k<inChannels*outChannels;k++) covL->connectModel[k]=true;

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

PoolLayer* initPoolLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels)
{
	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));

	poolL->inputHeight=inputHeight;
	poolL->inputWidth=inputWidth;
	poolL->mapSize=mapSize;
	poolL->inChannels=inChannels;
	poolL->outChannels=outChannels;

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
	float maxnum=vec[0];
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
		if(rbfPredict(&cnn->O7)!=vecmaxIndex(outputData->LabelPtr[n].LabelData,RBF_CLASSES))
			incorrectnum++;
		cnnclear(cnn);
	}
	return (float)incorrectnum/(float)testNum;
}

// 文件版本固定当前的缩放tanh/可训练下采样语义；结构变化时拒绝误加载。
static const char model_magic[8]={'D','L','C','R','B','F','0','1'};

static void model_shape(const CNN* cnn,int shape[17])
{
	shape[0]=cnn->C1->inputWidth;
	shape[1]=cnn->C1->inputHeight;
	shape[2]=cnn->C1->inChannels;
	shape[3]=cnn->C1->outChannels;
	shape[4]=cnn->C1->mapSize;
	shape[5]=cnn->S2->mapSize;
	shape[6]=cnn->C3->inChannels;
	shape[7]=cnn->C3->outChannels;
	shape[8]=cnn->C3->mapSize;
	shape[9]=cnn->S4->mapSize;
	shape[10]=RBF_INPUTS;
	shape[11]=RBF_CLASSES;
	shape[12]=cnn->C5->inChannels;
	shape[13]=cnn->C5->outChannels;
	shape[14]=cnn->C5->mapSize;
	shape[15]=cnn->F6->inputNum;
	shape[16]=cnn->F6->outputNum;
}

// 通道连接按输入优先存为0/1字节；加载时核对当前连接规则。
static void write_connections(const CovLayer* layer,FILE* fp)
{
	for(int i=0;i<layer->inChannels;i++)
		for(int j=0;j<layer->outChannels;j++) {
			unsigned char value=covConnected(layer,i,j) ? 1:0;
			write_exact(&value,1,1,fp);
		}
}

static void check_connections(const CovLayer* layer,FILE* fp)
{
	for(int i=0;i<layer->inChannels;i++)
		for(int j=0;j<layer->outChannels;j++) {
			unsigned char value;
			read_exact(&value,1,1,fp);
			if(value != (unsigned char)covConnected(layer,i,j)) {
				fprintf(stderr,"Model connection table does not match the current CNN\n");
				fclose(fp); exit(EXIT_FAILURE);
			}
		}
}

// 保存cnn
// 固定模板也写入并核对，避免不同字模或损失设置静默加载同一权重。
static void write_rbf_config(FILE* fp)
{
	const float reject_cost=(float)RBF_REJECT_COST;
	write_exact(&reject_cost,sizeof(float),1,fp);
	for(int j=0;j<RBF_CLASSES;j++) for(int k=0;k<RBF_INPUTS;k++) {
		unsigned char bit=rbfCenter(j,k)>0 ? 1:0;
		write_exact(&bit,1,1,fp);
	}
}
static void check_rbf_config(FILE* fp)
{
	float reject_cost;
	read_exact(&reject_cost,sizeof(float),1,fp);
	if(reject_cost!=(float)RBF_REJECT_COST) {
		fprintf(stderr,"Model RBF loss configuration does not match\n");exit(EXIT_FAILURE);
	}
	for(int j=0;j<RBF_CLASSES;j++) for(int k=0;k<RBF_INPUTS;k++) {
		unsigned char bit;
		read_exact(&bit,1,1,fp);
		if(bit!=(unsigned char)(rbfCenter(j,k)>0)) {
			fprintf(stderr,"Model RBF templates do not match\n");exit(EXIT_FAILURE);
		}
	}
}

void savecnn(CNN* cnn, const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}

	int shape[17];
	model_shape(cnn,shape);
	write_exact(model_magic,1,sizeof(model_magic),fp);
	write_exact(shape,sizeof(int),17,fp);
	write_connections(cnn->C1,fp);
	write_connections(cnn->C3,fp);
	write_connections(cnn->C5,fp);
	write_rbf_config(fp);

	int i,j,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			if(covConnected(cnn->C1,i,j))
			for(r=0;r<cnn->C1->mapSize;r++)
				write_exact(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	write_exact(cnn->C1->basicData,sizeof(float),cnn->C1->outChannels,fp);

	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			if(covConnected(cnn->C3,i,j))
			for(r=0;r<cnn->C3->mapSize;r++)
				write_exact(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	write_exact(cnn->C3->basicData,sizeof(float),cnn->C3->outChannels,fp);

	// C5网络
	for(i=0;i<cnn->C5->inChannels;i++)
		for(j=0;j<cnn->C5->outChannels;j++)
			if(covConnected(cnn->C5,i,j))
			for(r=0;r<cnn->C5->mapSize;r++)
				write_exact(cnn->C5->mapData[i][j][r],sizeof(float),cnn->C5->mapSize,fp);

	write_exact(cnn->C5->basicData,sizeof(float),cnn->C5->outChannels,fp);

	// F6全连接层。
	for(i=0;i<cnn->F6->outputNum;i++)
		write_exact(cnn->F6->wData[i],sizeof(float),cnn->F6->inputNum,fp);
	write_exact(cnn->F6->basicData,sizeof(float),cnn->F6->outputNum,fp);

	write_exact(cnn->S2->scaleData,sizeof(float),cnn->S2->outChannels,fp);
	write_exact(cnn->S2->basicData,sizeof(float),cnn->S2->outChannels,fp);
	write_exact(cnn->S4->scaleData,sizeof(float),cnn->S4->outChannels,fp);
	write_exact(cnn->S4->basicData,sizeof(float),cnn->S4->outChannels,fp);

	if(fclose(fp) != 0) { perror(filename); exit(EXIT_FAILURE); }
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

	char magic[8];
	int expected[17],actual[17];
	read_exact(magic,1,sizeof(magic),fp);
	if(memcmp(magic,model_magic,sizeof(magic)) != 0) {
		fprintf(stderr,"Incompatible model: expected DLCRBF01; retrain for the new network\n");
		fclose(fp); exit(EXIT_FAILURE);
	}
	model_shape(cnn,expected);
	read_exact(actual,sizeof(int),17,fp);
	if(memcmp(actual,expected,sizeof(actual)) != 0) {
		fprintf(stderr,"Model architecture does not match the current CNN\n");
		fclose(fp); exit(EXIT_FAILURE);
	}

	check_connections(cnn->C1,fp);
	check_connections(cnn->C3,fp);
	check_connections(cnn->C5,fp);
	check_rbf_config(fp);
	CovLayerConnect(cnn->C1,cnn->C1->connectModel);
	CovLayerConnect(cnn->C3,cnn->C3->connectModel);
	CovLayerConnect(cnn->C5,cnn->C5->connectModel);

	int i,j,c,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			if(covConnected(cnn->C1,i,j))
			for(r=0;r<cnn->C1->mapSize;r++)
				for(c=0;c<cnn->C1->mapSize;c++){
					read_exact(&cnn->C1->mapData[i][j][r][c],sizeof(float),1,fp);
				}

	for(i=0;i<cnn->C1->outChannels;i++)
		read_exact(&cnn->C1->basicData[i],sizeof(float),1,fp);

	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			if(covConnected(cnn->C3,i,j))
			for(r=0;r<cnn->C3->mapSize;r++)
				for(c=0;c<cnn->C3->mapSize;c++)
				read_exact(&cnn->C3->mapData[i][j][r][c],sizeof(float),1,fp);

	for(i=0;i<cnn->C3->outChannels;i++)
		read_exact(&cnn->C3->basicData[i],sizeof(float),1,fp);

	// C5网络
	for(i=0;i<cnn->C5->inChannels;i++)
		for(j=0;j<cnn->C5->outChannels;j++)
			if(covConnected(cnn->C5,i,j))
			for(r=0;r<cnn->C5->mapSize;r++)
				for(c=0;c<cnn->C5->mapSize;c++)
				read_exact(&cnn->C5->mapData[i][j][r][c],sizeof(float),1,fp);

	for(i=0;i<cnn->C5->outChannels;i++)
		read_exact(&cnn->C5->basicData[i],sizeof(float),1,fp);

	// F6全连接层。
	for(i=0;i<cnn->F6->outputNum;i++)
		read_exact(cnn->F6->wData[i],sizeof(float),cnn->F6->inputNum,fp);
	read_exact(cnn->F6->basicData,sizeof(float),cnn->F6->outputNum,fp);

	read_exact(cnn->S2->scaleData,sizeof(float),cnn->S2->outChannels,fp);
	read_exact(cnn->S2->basicData,sizeof(float),cnn->S2->outChannels,fp);
	read_exact(cnn->S4->scaleData,sizeof(float),cnn->S4->outChannels,fp);
	read_exact(cnn->S4->basicData,sizeof(float),cnn->S4->outChannels,fp);
	if(fgetc(fp) != EOF) {
		fprintf(stderr,"Unexpected trailing model data\n");
		fclose(fp); exit(EXIT_FAILURE);
	}

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
			double l=cnnloss(cnn,outputData->LabelPtr[n].LabelData);
			cnnbp(cnn,outputData->LabelPtr[n].LabelData); // 后向传播，这里主要计算各神经元的误差梯度


			cnnapplygrads(cnn,opts,inputData->ImgPtr[n].ImgData); // 更新权重

			cnnclear(cnn);
			// 显示与反向传播一致的MAP损失，保留指数平滑。
			if(n==0)
				cnn->L[n]=(float)l;
			else
				cnn->L[n]=cnn->L[n-1]*0.99f+0.01f*(float)l;
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
// C5覆盖S4的整个5×5空间。直接用反向下标做数学卷积，避免临时旋转数组。
static void c5ff(CNN* cnn)
{
	CovLayer* layer=cnn->C5;
	int width=layer->inputWidth-layer->mapSize+1;
	int height=layer->inputHeight-layer->mapSize+1;
	for(int o=0;o<layer->outChannels;o++)
		for(int r=0;r<height;r++)
			for(int c=0;c<width;c++) {
				float sum=0.0f;
				for(int i=0;i<layer->inChannels;i++) {
					if(!covConnected(layer,i,o)) continue;
					for(int a=0;a<layer->mapSize;a++)
						for(int b=0;b<layer->mapSize;b++)
							sum+=layer->mapData[i][o][a][b]*
								cnn->S4->y[i][r+layer->mapSize-1-a][c+layer->mapSize-1-b];
				}
				layer->v[o][r][c]=sum;
				layer->y[o][r][c]=activation_Sigma(sum,layer->basicData[o]);
			}
}

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
			if(!covConnected(cnn->C1,j,i)) continue;
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

	// C3部分连接：仅汇总连接表中指定的S2通道。
	outSize.c=cnn->S4->inputWidth;
	outSize.r=cnn->S4->inputHeight;
	inSize.c=cnn->C3->inputWidth;
	inSize.r=cnn->C3->inputHeight;
	mapSize.c=cnn->C3->mapSize;
	mapSize.r=cnn->C3->mapSize;
	for(i=0;i<(cnn->C3->outChannels);i++){
		for(j=0;j<(cnn->C3->inChannels);j++){
			if(!covConnected(cnn->C3,j,i)) continue;
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
	c5ff(cnn);
	outSize.c=cnn->C5->inputWidth-cnn->C5->mapSize+1;
	outSize.r=cnn->C5->inputHeight-cnn->C5->mapSize+1;

	// C5展平后输入F6，nnff包含偏置，激活时不再重复添加。
	float* F6inData=(float*)malloc(cnn->F6->inputNum*sizeof(float));
	for(i=0;i<cnn->C5->outChannels;i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				F6inData[i*outSize.r*outSize.c+r*outSize.c+c]=cnn->C5->y[i][r][c];
	nSize f6Size={cnn->F6->inputNum,cnn->F6->outputNum};
	nnff(cnn->F6->v,F6inData,cnn->F6->wData,cnn->F6->basicData,f6Size);
	for(i=0;i<cnn->F6->outputNum;i++)
		cnn->F6->y[i]=activation_Sigma(cnn->F6->v[i],0.0f);
	free(F6inData);
	rbfff(&cnn->O7,cnn->F6->y);
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

double cnnloss(const CNN* cnn,const float* outputData)
{
	int target=0;
	for(int j=1;j<RBF_CLASSES;j++) if(outputData[j]>outputData[target]) target=j;
	return rbfloss(&cnn->O7,target);
}

void cnnbp(CNN* cnn,float* outputData) // 网络的后向传播
{
	int i,j,c,r;
	int target=vecmaxIndex(outputData,RBF_CLASSES);
	// RBF -> F6：先得到dL/d(F6.y)，再乘激活导数得到dL/d(F6.z)。
	rbfbp(&cnn->O7,cnn->F6->y,target,cnn->F6->d);
	for(i=0;i<cnn->F6->outputNum;i++)
		cnn->F6->d[i]*=sigma_derivation(cnn->F6->y[i]);

	// F6 -> C5：先算dL/d(C5.y)，再乘C5的激活导数得到dL/d(C5.z)。
	nSize outSize={cnn->C5->inputWidth-cnn->C5->mapSize+1,
		cnn->C5->inputHeight-cnn->C5->mapSize+1};
	for(i=0;i<cnn->C5->outChannels;i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++) {
				float dy=0.0f;
				int k=i*outSize.r*outSize.c+r*outSize.c+c;
				for(j=0;j<cnn->F6->outputNum;j++)
					dy+=cnn->F6->d[j]*cnn->F6->wData[j][k];
				cnn->C5->d[i][r][c]=dy*sigma_derivation(cnn->C5->y[i][r][c]);
			}

	// C5 -> S4：按前向的连接位置散加梯度，尚未乘S4激活导数。
	for(i=0;i<cnn->C5->inChannels;i++)
		for(j=0;j<cnn->C5->outChannels;j++) {
			if(!covConnected(cnn->C5,i,j)) continue;
			for(r=0;r<outSize.r;r++)
				for(c=0;c<outSize.c;c++)
					for(int a=0;a<cnn->C5->mapSize;a++)
						for(int b=0;b<cnn->C5->mapSize;b++)
							cnn->S4->d[i][r+cnn->C5->mapSize-1-a][c+cnn->C5->mapSize-1-b]+=
								cnn->C5->d[j][r][c]*cnn->C5->mapData[i][j][a][b];
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
	// 只沿C3的60组有效通道连接，将梯度传回S2。
	outSize.c=cnn->C3->inputWidth;
	outSize.r=cnn->C3->inputHeight;
	nSize inSize={cnn->S4->inputWidth,cnn->S4->inputHeight};
	nSize mapSize={cnn->C3->mapSize,cnn->C3->mapSize};
	for(i=0;i<cnn->S2->outChannels;i++){
		for(j=0;j<cnn->C3->outChannels;j++){
			if(!covConnected(cnn->C3,i,j)) continue;
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
			if(!covConnected(cnn->C1,j,i)) continue;
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
			if(!covConnected(cnn->C3,j,i)) continue;
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

	// C5：dL/dW=各输出位置的d乘对应S4输入之和；偏置梯度为d之和。
	nSize C5out={cnn->C5->inputWidth-cnn->C5->mapSize+1,
		cnn->C5->inputHeight-cnn->C5->mapSize+1};
	for(i=0;i<cnn->C5->outChannels;i++) {
		for(j=0;j<cnn->C5->inChannels;j++) {
			if(!covConnected(cnn->C5,j,i)) continue;
			for(int a=0;a<cnn->C5->mapSize;a++)
				for(int b=0;b<cnn->C5->mapSize;b++) {
					float grad=0.0f;
					for(r=0;r<C5out.r;r++)
						for(c=0;c<C5out.c;c++)
							grad+=cnn->C5->d[i][r][c]*
								cnn->S4->y[j][r+cnn->C5->mapSize-1-a][c+cnn->C5->mapSize-1-b];
					cnn->C5->dmapData[j][i][a][b]=grad;
					cnn->C5->mapData[j][i][a][b]-=opts.alpha*grad;
				}
		}
		cnn->C5->basicData[i]-=opts.alpha*summat(cnn->C5->d[i],C5out);
	}

	// F6权重梯度为d乘C5输出，偏置梯度就是d。
	for(j=0;j<cnn->F6->outputNum;j++) {
		for(i=0;i<cnn->C5->outChannels;i++)
			for(r=0;r<C5out.r;r++)
				for(c=0;c<C5out.c;c++) {
					int k=i*C5out.r*C5out.c+r*C5out.c+c;
					cnn->F6->wData[j][k]-=opts.alpha*cnn->F6->d[j]*cnn->C5->y[i][r][c];
				}
		cnn->F6->basicData[j]-=opts.alpha*cnn->F6->d[j];
	}
	poolapplygrads(cnn->S2,opts.alpha);
	poolapplygrads(cnn->S4,opts.alpha);
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
	// C5的特征图和局部梯度；参数梯度在每次更新时覆盖。
	for(j=0;j<cnn->C5->outChannels;j++)
		for(r=0;r<cnn->C5->inputHeight-cnn->C5->mapSize+1;r++)
			for(c=0;c<cnn->C5->inputWidth-cnn->C5->mapSize+1;c++) {
				cnn->C5->v[j][r][c]=0.0f;
				cnn->C5->y[j][r][c]=0.0f;
				cnn->C5->d[j][r][c]=0.0f;
			}
	// F6中间结果每个样本结束后清零。
	for(j=0;j<cnn->F6->outputNum;j++) {
		cnn->F6->d[j]=0.0f;
		cnn->F6->v[j]=0.0f;
		cnn->F6->y[j]=0.0f;
	}
	memset(&cnn->O7,0,sizeof(cnn->O7));

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

	// C5网络：权重、偏置及v/d/y。
	for(i=0;i<cnn->C5->inChannels;i++)
		for(j=0;j<cnn->C5->outChannels;j++)
			for(r=0;r<cnn->C5->mapSize;r++)
				write_exact(cnn->C5->mapData[i][j][r],sizeof(float),cnn->C5->mapSize,fp);
	write_exact(cnn->C5->basicData,sizeof(float),cnn->C5->outChannels,fp);
	for(j=0;j<cnn->C5->outChannels;j++)
		for(r=0;r<cnn->C5->inputHeight-cnn->C5->mapSize+1;r++) {
			int width=cnn->C5->inputWidth-cnn->C5->mapSize+1;
			write_exact(cnn->C5->v[j][r],sizeof(float),width,fp);
			write_exact(cnn->C5->d[j][r],sizeof(float),width,fp);
			write_exact(cnn->C5->y[j][r],sizeof(float),width,fp);
		}

	// F6权重、偏置及中间结果。
	for(i=0;i<cnn->F6->outputNum;i++)
		write_exact(cnn->F6->wData[i],sizeof(float),cnn->F6->inputNum,fp);
	write_exact(cnn->F6->basicData,sizeof(float),cnn->F6->outputNum,fp);
	write_exact(cnn->F6->v,sizeof(float),cnn->F6->outputNum,fp);
	write_exact(cnn->F6->d,sizeof(float),cnn->F6->outputNum,fp);
	write_exact(cnn->F6->y,sizeof(float),cnn->F6->outputNum,fp);

	// 固定RBF配置、距离和局部梯度。
	write_rbf_config(fp);
	write_exact(cnn->O7.y,sizeof(double),RBF_CLASSES,fp);
	write_exact(cnn->O7.d,sizeof(double),RBF_CLASSES,fp);

	fclose(fp);
}
