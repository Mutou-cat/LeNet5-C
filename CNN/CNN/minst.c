#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "minst.h"

// 按内存中的大端字节顺序还原IDX头字段；当前读取方式要求int为32位。
int ReverseInt(int i)
{
	unsigned char *split = (unsigned char*)&i;
	return ((int)split[0])<<24 | split[1]<<16 | split[2]<<8 | split[3];
}

ImgArr read_Img(const char* filename) // 读取28×28的MNIST图像，返回32×32的LeNet-5输入
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	// IDX头字段为4字节大端整数：先读取原始字节，再还原数值。
	read_exact((char*)&magic_number,sizeof(magic_number),1,fp);
	magic_number = ReverseInt(magic_number);
	// 图片样本数，决定随后读取多少组784字节。
	read_exact((char*)&number_of_images,sizeof(number_of_images),1,fp);
	number_of_images = ReverseInt(number_of_images);
	// 文件中每张图片的行数。
	read_exact((char*)&n_rows,sizeof(n_rows),1,fp);
	n_rows = ReverseInt(n_rows);
	// 文件中每张图片的列数。
	read_exact((char*)&n_cols,sizeof(n_cols),1,fp);
	n_cols = ReverseInt(n_cols);
	if(magic_number != 2051 || number_of_images <= 0 ||
	   n_rows != 28 || n_cols != 28) {
		fprintf(stderr,"Expected a nonempty 28x28 MNIST image file: %s\n",filename);
		fclose(fp);
		exit(EXIT_FAILURE);
	}

	// 文件里的尺寸仍为28×28；内存中四周各扩展2个背景像素。
	const int padding = 2;
	const int image_rows = n_rows + 2 * padding;
	const int image_cols = n_cols + 2 * padding;
	int i,r,c;

	// 分配容器和图片结构体数组；后续再为每张图分配行指针及像素。
	ImgArr imgarr=(ImgArr)malloc(sizeof(MinstImgArr));
	if(imgarr == NULL) { perror("malloc"); exit(EXIT_FAILURE); }
	imgarr->ImgNum=number_of_images;
	imgarr->ImgPtr=(MinstImg*)malloc(number_of_images*sizeof(MinstImg));
	if(imgarr->ImgPtr == NULL) { perror("malloc"); exit(EXIT_FAILURE); }

	for(i = 0; i < number_of_images; ++i)
	{
		imgarr->ImgPtr[i].r=image_rows;
		imgarr->ImgPtr[i].c=image_cols;
		imgarr->ImgPtr[i].ImgData=(float**)malloc(image_rows*sizeof(float*));
		if(imgarr->ImgPtr[i].ImgData == NULL) { perror("malloc"); exit(EXIT_FAILURE); }
		for(r = 0; r < image_rows; ++r) {
			imgarr->ImgPtr[i].ImgData[r]=(float*)malloc(image_cols*sizeof(float));
			if(imgarr->ImgPtr[i].ImgData[r] == NULL) { perror("malloc"); exit(EXIT_FAILURE); }
			for(c = 0; c < image_cols; ++c)
				imgarr->ImgPtr[i].ImgData[r][c]=-0.1f;
		}

		// 每张图只读文件中原有的784个像素，避免读取位置错位。
		for(r = 0; r < n_rows; ++r)
		{
			for(c = 0; c < n_cols; ++c)
			{
				unsigned char temp = 0;
				read_exact((char*) &temp, sizeof(temp),1,fp);
				// 背景0映射为-0.1，笔画255映射为1.175。
				imgarr->ImgPtr[i].ImgData[r+padding][c+padding]=
					-0.1f+1.275f*((float)temp/255.0f);
			}
		}
	}

	fclose(fp);
	return imgarr;
}

LabelArr read_Lable(const char* filename)// 读取标签：8字节头后，每个样本占1字节
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}
	assert(fp); // 上面的失败分支已退出；此处是冗余的调试断言。

	int magic_number = 0;
	int number_of_labels = 0;
	int label_long = 10;

	// IDX头字段为4字节大端整数：先读取原始字节，再还原数值。
	read_exact((char*)&magic_number,sizeof(magic_number),1,fp);
	magic_number = ReverseInt(magic_number); // 当前仅读取，未校验标签魔数2049。
	// 标签样本数；调用方核对它与图片数量是否一致。
	read_exact((char*)&number_of_labels,sizeof(number_of_labels),1,fp);
	number_of_labels = ReverseInt(number_of_labels);

	int i;

	// 分配标签容器；每个标签再分配一个10维one-hot向量。
	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabelArr));
	labarr->LabelNum=number_of_labels;
	labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));

	for(i = 0; i < number_of_labels; ++i)
	{
		labarr->LabelPtr[i].l=10;
		// 先置零，再把文件中数字temp对应的位置设为1。
		labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float));
		unsigned char temp = 0;
		read_exact((char*) &temp, sizeof(temp),1,fp);
		if (temp >= 10) { fprintf(stderr, "Invalid MNIST label\n"); exit(EXIT_FAILURE); }
		labarr->LabelPtr[i].LabelData[(int)temp]=1.0;
	}

	fclose(fp);
	return labarr;
}

char* intTochar(int i)// 将非负整数转为十进制字符串，返回堆内存
{
	int itemp=i;
	int w=0;
	while(itemp>=10){
		itemp=itemp/10;
		w++;
	}
	char* ptr=(char*)malloc((w+2)*sizeof(char));
	ptr[w+1]='\0';
	int r; // 余数
	while(i>=10){
		r=i%10;
		i=i/10;
		ptr[w]=(char)(r+48);
		w--;
	}
	ptr[w]=(char)(i+48);
	return ptr;
}

char * combine_strings(const char *a, const char *b) // 拼接字符串，返回新分配的缓冲区；调用方负责释放
{
	char *ptr;
	int lena=strlen(a),lenb=strlen(b);
	int i,l=0;
	ptr = (char *)malloc((lena+lenb+1) * sizeof(char));
	for(i=0;i<lena;i++)
		ptr[l++]=a[i];
	for(i=0;i<lenb;i++)
		ptr[l++]=b[i];
	ptr[l]='\0';
	return(ptr);
}

// .gray里保存的是内存中的float像素，不含文件头，不能作为predict的784字节输入。
void save_Img(ImgArr imgarr,const char* filedir) // 历史导出工具：filedir为路径前缀，需自带目录分隔符，目录须已存在
{
	int img_number=imgarr->ImgNum;

	int i,r;
	for(i=0;i<img_number;i++){
		const char* filename=combine_strings(filedir,combine_strings(intTochar(i),".gray"));
		FILE  *fp=NULL;
		fp=fopen(filename,"wb");
		if(fp==NULL)
			printf("write file failed\n");
		assert(fp);

		for(r=0;r<imgarr->ImgPtr[i].r;r++)
			fwrite(imgarr->ImgPtr[i].ImgData[r],sizeof(float),imgarr->ImgPtr[i].c,fp);

		fclose(fp);
	}
}
