#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "minst.h"

//英特尔处理器和其他低端机用户必须翻转头字节。
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
	//从文件中读取sizeof(magic_number) 个字符到 &magic_number
	read_exact((char*)&magic_number,sizeof(magic_number),1,fp);
	magic_number = ReverseInt(magic_number);
	//获取训练或测试image的个数number_of_images
	read_exact((char*)&number_of_images,sizeof(number_of_images),1,fp);
	number_of_images = ReverseInt(number_of_images);
	//获取训练或测试图像的高度Heigh
	read_exact((char*)&n_rows,sizeof(n_rows),1,fp);
	n_rows = ReverseInt(n_rows);
	//获取训练或测试图像的宽度Width
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

	// 图像数组的初始化
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

LabelArr read_Lable(const char* filename)// 读入图像
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}
	assert(fp);

	int magic_number = 0;
	int number_of_labels = 0;
	int label_long = 10;

	//从文件中读取sizeof(magic_number) 个字符到 &magic_number
	read_exact((char*)&magic_number,sizeof(magic_number),1,fp);
	magic_number = ReverseInt(magic_number);
	//获取训练或测试image的个数number_of_images
	read_exact((char*)&number_of_labels,sizeof(number_of_labels),1,fp);
	number_of_labels = ReverseInt(number_of_labels);

	int i;

	// 图像标记数组的初始化
	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabelArr));
	labarr->LabelNum=number_of_labels;
	labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));

	for(i = 0; i < number_of_labels; ++i)
	{
		labarr->LabelPtr[i].l=10;
		labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float));
		unsigned char temp = 0;
		read_exact((char*) &temp, sizeof(temp),1,fp);
		if (temp >= 10) { fprintf(stderr, "Invalid MNIST label\n"); exit(EXIT_FAILURE); }
		labarr->LabelPtr[i].LabelData[(int)temp]=1.0;
	}

	fclose(fp);
	return labarr;
}

char* intTochar(int i)// 将数字转换成字符串
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

char * combine_strings(const char *a, const char *b) // 将两个字符串相连
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

void save_Img(ImgArr imgarr,const char* filedir) // 将图像数据保存成文件
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
