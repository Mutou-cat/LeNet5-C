#define _POSIX_C_SOURCE 200809L
#include <stdarg.h>
#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "cnn.h"

/* 阅读约定：x为本层输入，z为加偏置后的线性结果，y=f(z)。
 * 卷积/全连接层的d是当前样本的dL/dz，不是参数，也不是参数更新量。
 * 前向顺序：C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> O7；反传顺序相反。
 * cnnbp先传播局部梯度；cnnapplygrads再结合缓存的输入求参数梯度并执行SGD。
 */

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

// 拷贝连接表并清零断开槽；不重采样权重。加载模型时也用它恢复无效槽为0。
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

// LeCun等(1998)附录A：U[-2.4/F, 2.4/F]，F为神经元的输入连接数。
// srand由程序入口调用一次，各层连续使用同一个随机序列。
static float initial_weight(int fan_in)
{
	float uniform=(float)((double)rand()/(double)RAND_MAX*2.0-1.0);
	return uniform*(2.4f/(float)fan_in);
}

static CovLayer* initCovLayerConnected(int inputWidth,int inputHeight,int mapSize,
	int inChannels,int outChannels,const bool* connections);

// 当前网络面向32×32单通道输入、10类输出；inputSize按{宽,高}传入。
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
	// 位掩码按输出通道组织，运行时转换为[输入*16+输出]的一维连接表。
	bool connections[6*16];
	for(int i=0;i<6;i++)
		for(int j=0;j<16;j++)
			connections[i*16+j]=(c3_connections[j] & (1u<<i)) != 0;
	cnn->C3=initCovLayerConnected(inSize.c,inSize.r,5,6,16,connections);
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

static CovLayer* initCovLayerConnected(int inputWidth,int inputHeight,int mapSize,
	int inChannels,int outChannels,const bool* connections)
{
	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));

	covL->inputHeight=inputHeight;
	covL->inputWidth=inputWidth;
	covL->mapSize=mapSize;

	covL->inChannels=inChannels;
	covL->outChannels=outChannels;

	covL->isFullConnect=true;
	covL->connectModel=(bool*)malloc(inChannels*outChannels*sizeof(bool));
	for(int k=0;k<inChannels*outChannels;k++) {
		covL->connectModel[k]=connections ? connections[k]:true;
		if(!covL->connectModel[k]) covL->isFullConnect=false;
	}

	// 按[输入通道][输出通道][核行][核列]分配；各行不保证连续存储。
	int i,j,c,r;
	covL->mapData=(float****)malloc(inChannels*sizeof(float***));
	for(i=0;i<inChannels;i++){
		covL->mapData[i]=(float***)malloc(outChannels*sizeof(float**));
		for(j=0;j<outChannels;j++){
			// fan-in按当前输出实际连接的通道计数，C3各输出可能不同。
			int connected_inputs=0;
			for(int k=0;k<inChannels;k++)
				connected_inputs+=covConnected(covL,k,j);
			int fan_in=mapSize*mapSize*connected_inputs;
			covL->mapData[i][j]=(float**)malloc(mapSize*sizeof(float*));
			for(r=0;r<mapSize;r++){
				covL->mapData[i][j][r]=(float*)malloc(mapSize*sizeof(float));
				for(c=0;c<mapSize;c++){
					covL->mapData[i][j][r][c]=covConnected(covL,i,j) ? initial_weight(fan_in):0.0f;
				}
			}
		}
	}
	// 与权重同形的缓存：当前C5写入dL/dW，C1/C3直接使用临时梯度矩阵。
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

	// 每输出通道共享一个偏置；零初始化是本实现的约定。
	covL->basicData=(float*)calloc(outChannels,sizeof(float));

	// valid卷积、步长1；v/y/d是输出特征图大小，不是卷积核大小。
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

// 输入输出通道全部相连的卷积层走同一路径；部分连接须在初始化前提供掩码。
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels)
{
	return initCovLayerConnected(inputWidth,inputHeight,mapSize,inChannels,outChannels,NULL);
}

// 输入输出通道一一对应；本网络窗口为2×2、步长2，每通道只有a和b两个参数。
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
	// a在窗口内各连接之间共享，按窗口面积作为fan-in随机初始化；不是固定平均系数。
	for(int channel=0;channel<outChannels;channel++)
		poolL->scaleData[channel]=initial_weight(mapSize*mapSize);

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
	outL->wData=(float**)malloc(outputNum*sizeof(float*)); // 行对应输出神经元，列对应输入分量
	int i,j;
	for(i=0;i<outputNum;i++){
		outL->wData[i]=(float*)malloc(inputNum*sizeof(float));
		for(j=0;j<inputNum;j++){
			outL->wData[i][j]=initial_weight(inputNum);
		}
	}

	outL->isFullConnect=true;

	return outL;
}

int vecmaxIndex(float* vec, int veclength)// 返回首个最大值下标，要求vec非空且veclength>0；用于取one-hot目标类别
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

// 对前testNum个样本评估，RBF预测取最小距离；返回错误率而非准确率。
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

// DLCRBF01模型格式；整数和浮点使用本机表示，不是跨平台交换协议。
static const char model_magic[8]={'D','L','C','R','B','F','0','1'};

// 17个字段的顺序属于文件协议；修改字段含义或布局时须同步升级版本。
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

/* 文件顺序：版本、结构、C1/C3/C5连接表、RBF配置、C1/C3/C5/F6参数、S2/S4参数。
 * 仅保存有效卷积连接；不保存v/y/d、损失曲线或随机数状态，不能据此恢复训练进度。 */
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
	// C1卷积核和偏置，核按输入通道优先写入/读取。
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			if(covConnected(cnn->C1,i,j))
			for(r=0;r<cnn->C1->mapSize;r++)
				write_exact(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	write_exact(cnn->C1->basicData,sizeof(float),cnn->C1->outChannels,fp);

	// C3：仅对60组有效连接处理权重，另有16个偏置。
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			if(covConnected(cnn->C3,i,j))
			for(r=0;r<cnn->C3->mapSize;r++)
				write_exact(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	write_exact(cnn->C3->basicData,sizeof(float),cnn->C3->outChannels,fp);

	// C5卷积核和偏置。
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
// 先校验结构及固定配置，再写入已分配的网络；与savecnn的顺序严格对应。
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
	// 文件不保存断开槽，先将这些槽清零；有效权重稍后从文件覆盖。
	CovLayerConnect(cnn->C1,cnn->C1->connectModel);
	CovLayerConnect(cnn->C3,cnn->C3->connectModel);
	CovLayerConnect(cnn->C5,cnn->C5->connectModel);

	int i,j,c,r;
	// C1卷积核和偏置，核按输入通道优先写入/读取。
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			if(covConnected(cnn->C1,i,j))
			for(r=0;r<cnn->C1->mapSize;r++)
				for(c=0;c<cnn->C1->mapSize;c++){
					read_exact(&cnn->C1->mapData[i][j][r][c],sizeof(float),1,fp);
				}

	for(i=0;i<cnn->C1->outChannels;i++)
		read_exact(&cnn->C1->basicData[i],sizeof(float),1,fp);

	// C3：仅对60组有效连接处理权重，另有16个偏置。
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			if(covConnected(cnn->C3,i,j))
			for(r=0;r<cnn->C3->mapSize;r++)
				for(c=0;c<cnn->C3->mapSize;c++)
				read_exact(&cnn->C3->mapData[i][j][r][c],sizeof(float),1,fp);

	for(i=0;i<cnn->C3->outChannels;i++)
		read_exact(&cnn->C3->basicData[i],sizeof(float),1,fp);

	// C5卷积核和偏置。
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

// 单调时钟测量真实耗时，不受系统时间校准影响；不是CPU占用时间。
static double training_seconds(void)
{
	struct timespec now;
	if(clock_gettime(CLOCK_MONOTONIC,&now)!=0) { perror("clock_gettime");exit(EXIT_FAILURE); }
	return (double)now.tv_sec+(double)now.tv_nsec/1e9;
}

// 同时输出到终端和可选日志文件，每次刷新，便于实时查看和保留中断前的进度。
static void training_log(FILE* log,const char* format,...)
{
	va_list args;
	va_start(args,format);
	vprintf(format,args);
	va_end(args);
	fflush(stdout);
	if(log) {
		// vprintf已经消费过变参列表，写日志前须重新va_start。
		va_start(args,format);
		int result=vfprintf(log,format,args);
		va_end(args);
		if(result<0 || fflush(log)!=0) { perror("training log");exit(EXIT_FAILURE); }
	}
}

void cnntrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum,FILE* log)
{
	// 计时覆盖本函数内的计算与日志I/O，不包含调用前读数据、初始化和调用后存模型。
	double training_start=training_seconds();
	time_t now=time(NULL);
	char started[80]="unknown";
	struct tm* local=localtime(&now);
	if(local) strftime(started,sizeof(started),"%Y-%m-%d %H:%M:%S %z",local);
	training_log(log,"\nTraining started: %s\n",started);
	// 每轮复用这段损失记录；L[0]直接取首个样本，后续为0.99/0.01指数平滑。
	cnn->L=(float*)malloc(trainNum*sizeof(float));
	training_log(log,"Training: %d samples, %d epochs, learning rate %.4g\n",
		trainNum, opts.numepochs, opts.alpha);
	fflush(stdout);
	int e;
	for(e=0;e<opts.numepochs;e++){
		double epoch_start=training_seconds();
		training_log(log,"Epoch %d/%d: starting\n", e+1, opts.numepochs);
		fflush(stdout);
		int n=0;
		// 固定文件顺序逐样本SGD；这里没有打乱、批量梯度或学习率衰减。
		for(n=0;n<trainNum;n++){
			cnnff(cnn,inputData->ImgPtr[n].ImgData);  // 缓存当前样本各层输出和RBF距离
			// 在更新与清空缓存之前取损失，确保日志对应本次前向结果。
			double l=cnnloss(cnn,outputData->LabelPtr[n].LabelData);
			cnnbp(cnn,outputData->LabelPtr[n].LabelData); // 用旧权重计算各层d及下采样参数梯度


			cnnapplygrads(cnn,opts,inputData->ImgPtr[n].ImgData); // 求卷积/全连接参数梯度，更新全部可训练参数

			cnnclear(cnn);
			// 平滑值仅用于观察趋势，不是整轮平均损失，也不参与反向传播。
			if(n==0)
				cnn->L[n]=(float)l;
			else
				cnn->L[n]=cnn->L[n-1]*0.99f+0.01f*(float)l;
			// 首个样本、每 100 个样本及本轮结束时输出平滑损失。
			if(n==0 || (n+1)%100==0 || n+1==trainNum){
				training_log(log,"Epoch %d/%d | %d/%d (%6.2f%%) | smoothed loss: %.6f | elapsed: %.1f s\n",
					e+1, opts.numepochs, n+1, trainNum,
					100.0*(n+1)/trainNum, cnn->L[n],training_seconds()-training_start);
				fflush(stdout);
			}
		}
		training_log(log,"Epoch %d/%d finished: %.3f s\n",e+1,opts.numepochs,
			training_seconds()-epoch_start);
	}
	double elapsed=training_seconds()-training_start;
	long long seconds=(long long)elapsed;
	training_log(log,"Training time: %.3f s (%02lld:%02lld:%02lld)\n",
		elapsed,seconds/3600,seconds/60%60,seconds%60);
}

/* C5覆盖S4的整个5×5空间，当前每个输出通道只有一个z。
 * 数学卷积：z[o,r,c]=b[o]+sum_{i,a,b} W[i,o,a,b]*x[i,r+K-1-a,c+K-1-b]。
 * 将翻转体现在输入下标上，与cov中旋转核等价，避免创建临时矩阵。
 */
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
				layer->v[o][r][c]=sum+layer->basicData[o];
				layer->y[o][r][c]=activation_Sigma(layer->v[o][r][c]);
			}
}

void cnnff(CNN* cnn,float** inputData)
{
	// C1：单张inputData[行][列]，32×32×1 -> 28×28×6。
	int i,j,r,c;
	// nSize是{宽,高}；卷积输出缓冲会累加，样本之间须调用cnnclear。
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
			for(c=0;c<outSize.c;c++) {
				cnn->C1->v[i][r][c]+=cnn->C1->basicData[i];
				cnn->C1->y[i][r][c]=activation_Sigma(cnn->C1->v[i][r][c]);
			}
	}

	// S2：窗口求和、乘每通道系数、加偏置、缩放tanh。
	poolff(cnn->S2,cnn->C1->y);

	// C3：14×14×6 -> 10×10×16；先累加相连通道的卷积，再加一次偏置并激活。
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
			for(c=0;c<outSize.c;c++) {
				cnn->C3->v[i][r][c]+=cnn->C3->basicData[i];
				cnn->C3->y[i][r][c]=activation_Sigma(cnn->C3->v[i][r][c]);
			}
	}

	// S4：10×10×16 -> 5×5×16；C5随后变为1×1×120。
	poolff(cnn->S4,cnn->C3->y);
	c5ff(cnn);
	outSize.c=cnn->C5->inputWidth-cnn->C5->mapSize+1;
	outSize.r=cnn->C5->inputHeight-cnn->C5->mapSize+1;

	// C5按通道、行、列展平：k=i*H*W+r*W+c；当前空间1×1，得到120维。
	// nnff已经加偏置，激活函数只处理z。
	float* F6inData=(float*)malloc(cnn->F6->inputNum*sizeof(float));
	for(i=0;i<cnn->C5->outChannels;i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				F6inData[i*outSize.r*outSize.c+r*outSize.c+c]=cnn->C5->y[i][r][c];
	nSize f6Size={cnn->F6->inputNum,cnn->F6->outputNum};
	nnff(cnn->F6->v,F6inData,cnn->F6->wData,cnn->F6->basicData,f6Size);
	for(i=0;i<cnn->F6->outputNum;i++)
		cnn->F6->y[i]=activation_Sigma(cnn->F6->v[i]);
	free(F6inData);
	// O7将84维特征与10个固定模板比较，输出10个距离，最小者为预测类别。
	rbfff(&cnn->O7,cnn->F6->y);
}

// Sigma是沿用的函数名；当前实现是缩放tanh，不是早期的Logistic sigmoid。
float activation_Sigma(float z) // LeNet-5缩放tanh激活函数
{
	const float A=1.7159f;
	const float S=2.0f/3.0f;
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
				// 输出(r,c)对应输入窗口左上角(r*K,c*K)，窗口之间不重叠。
				float sum=0.0f;
				for(int dr=0;dr<pool->mapSize;dr++)
					for(int dc=0;dc<pool->mapSize;dc++)
						sum+=input[i][r*pool->mapSize+dr][c*pool->mapSize+dc];
				pool->sumData[i][r][c]=sum;
				float z=pool->scaleData[i]*sum+pool->basicData[i];
				pool->y[i][r][c]=activation_Sigma(z);
			}
}


// 点积返回一个标量，不是逐元素乘积向量。
float vecMulti(float* vec1,float* vec2,int vecL)// 两向量相乘
{
	int i;
	float m=0;
	for(i=0;i<vecL;i++)
		m=m+vec1[i]*vec2[i];
	return m;
}

// output[j]=sum_i(wdata[j][i]*input[i])+bas[j]；仅线性运算，不执行激活。
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
	// dy/dz=A*S*(1-tanh(S*z)^2)，由缓存的y恢复tanh，无需重算z。
	return A*S*(1.0f-t*t);
}

/* 调用前d是dL/dy，调用后d是dL/dz；每个样本只调用一次，不能重复乘激活导数。
 * z=a*s+b，其中s为窗口和：dL/da=sum_{r,c}(d*s)，dL/db=sum_{r,c}d。
 * a、b在同一通道内共享，因此累加所有空间位置；输入梯度a*d由cnnbp继续传递。
 */
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

// outputData为10维one-hot；argmax得到数字编号，RBF不直接回归这10个标签值。
double cnnloss(const CNN* cnn,const float* outputData)
{
	int target=0;
	for(int j=1;j<RBF_CLASSES;j++) if(outputData[j]>outputData[target]) target=j;
	return rbfloss(&cnn->O7,target);
}

/* 当前样本必须已完成cnnff；所有累加缓冲在该样本开始前应为0。
 * 此函数求各层局部梯度及S2/S4的参数梯度，不求原始图片梯度。
 * C1/C3/C5/F6的权重、偏置梯度在cnnapplygrads中用这些d和前向输入计算。
 */
void cnnbp(CNN* cnn,float* outputData) // 网络的反向传播
{
	int i,j,c,r;
	int target=vecmaxIndex(outputData,RBF_CLASSES);
	// RBF -> F6：先得到dL/d(F6.y)，再乘激活导数得到dL/d(F6.z)。
	rbfbp(&cnn->O7,cnn->F6->y,target,cnn->F6->d);
	for(i=0;i<cnn->F6->outputNum;i++)
		cnn->F6->d[i]*=sigma_derivation(cnn->F6->y[i]);

	// F6 -> C5：z_F6[j]=sum_k W[j,k]*y_C5[k]+b[j]。
	// 因此dL/dy_C5[k]=sum_j F6.d[j]*W[j,k]；再乘f'(z_C5[k])。
	// k与前向展平顺序一致；每个C5输出收到全部84个F6神经元的贡献。
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

	// C5 -> S4：前向中某个x乘了W，反向就向该x的位置累加C5.d*W。
	// 这里使用与c5ff完全相同的翻转下标；累加120个输出通道的贡献。
	// 此时S4.d仍是dL/d(S4.y)，后面的poolbp才将其变成dL/d(S4.z)。
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
	// 将S4的dL/dz复制到对应2×2窗口，再乘S4的a及C3激活导数。
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
	// 前向z[r,c]=sum W[a,b]*x[r+K-1-a,c+K-1-b]，对x[u,v]求导得到
	// dx[u,v]=sum W[a,b]*d[u-K+1+a,v-K+1+b]（越界d视为0）。
	// 这正是下面的full互相关，不应再次翻转核；10+5-1=14。
	// 单次corr为14×14，一个S2通道还须累加所有与它相连的C3输出通道。
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

	// S2 -> C1：与S4 -> C3相同的链式法则，最终得到28×28×6个dL/dz。
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

void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData) // 计算参数梯度并更新权重、偏置及下采样系数
{
	// 所有d已用旧权重算完，再执行w-=alpha*dL/dw；更新C1/C3/C5/F6和S2/S4，RBF固定。
	int i,j,r,c;

	// C1的共享权重梯度累加所有输出位置：g[a][b]=sum(d[r][c]*x[r+K-1-a][c+K-1-b])。
	// 这里历史代码写成{高,宽}，仅因当前图片/特征图均为正方形才等价于{宽,高}。
	nSize dSize={cnn->S2->inputHeight,cnn->S2->inputWidth};
	nSize ySize={cnn->C1->inputHeight,cnn->C1->inputWidth};
	nSize mapSize={cnn->C1->mapSize,cnn->C1->mapSize};

	for(i=0;i<cnn->C1->outChannels;i++){
		for(j=0;j<cnn->C1->inChannels;j++){
			if(!covConnected(cnn->C1,j,i)) continue;
			// 以28×28的d作核、32×32的旋转输入作图，valid结果为5×5权重梯度。
			// cov再旋转d，换元后恰为上面的sum(d*x)公式，结果按原W[a][b]方向排列。
			// 两次旋转不是互相抵消；若直接对未翻转的x和d做相关，核梯度方向会不同。
			float** flipinputData=rotate180(inputData,ySize);
			float** C1dk=cov(cnn->C1->d[i],dSize,flipinputData,ySize,valid);
			// C1dk从梯度变为参数增量-alpha*g，再加回权重。
			multifactor(C1dk,C1dk,mapSize,-1*opts.alpha);
			addmat(cnn->C1->mapData[j][i],cnn->C1->mapData[j][i],mapSize,C1dk,mapSize);
			for(r=0;r<mapSize.r;r++)
				free(C1dk[r]);
			free(C1dk);
			for(r=0;r<ySize.r;r++)
				free(flipinputData[r]);
			free(flipinputData);
		}
		// 一个通道的偏置作用于全部输出位置，因此偏置梯度是该通道d之和。
		cnn->C1->basicData[i]=cnn->C1->basicData[i]-opts.alpha*summat(cnn->C1->d[i],dSize);
	}

	// C3同样求共享核梯度，但输入是S2.y，且只更新连接表中的有效核。
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

	// F6：dL/dW[j,k]=F6.d[j]*C5.y[k]，dL/db[j]=F6.d[j]。
	// 虽然C5权重已更新，此处仍使用本次前向缓存的旧C5.y，不重新前向。
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
	// 清理每样本的累加缓冲；不修改模型参数或L。C5的dmapData由下次更新覆盖。
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
	// C3的中间结果。
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

// 原项目保留的调试转储函数，当前CLI没有调用。
// 包含特征图和完整核槽，文件无模型版本头，不能交给importcnn读取。
void savecnndata(CNN* cnn,const char* filename,float** inputdata) // 导出当前输入、参数和中间结果供离线观察
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}

	// 当前输入图片。
	int i,j,r;
	// 先写输入图片，再写C1权重、偏置及v/d/y。
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

	// S2的d/y；此历史转储不包含S2的可训练参数。
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->d[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->y[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
	}
	// C3完整权重存储槽（含断开连接）、偏置及v/d/y。
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

	// S4的d/y；此历史转储不包含S4的可训练参数。
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
