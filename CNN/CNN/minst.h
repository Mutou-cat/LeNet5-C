#ifndef __MINST_
#define __MINST_
/* MNIST IDX接口，Minst/read_Lable保留原项目命名。
 * 图片与标签按下标对应；返回的容器、行数组和标签向量在堆上分配。
 * 读入后文件已关闭；数据仍有效。若释放，须先逐行/逐标签释放，再释放结构体数组及容器。 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

typedef struct MinstImg{
	int c;           // 内存图像宽度，读入后为32
	int r;           // 内存图像高度，读入后为32
	float** ImgData; // ImgData[行][列]，归一化并填边后的float像素
}MinstImg;

typedef struct MinstImgArr{
	int ImgNum;        // 存储图像的数目
	MinstImg* ImgPtr;  // 存储图像数组指针
} MinstImgArr, *ImgArr;              // ImgArr是MinstImgArr的指针别名

typedef struct MinstLabel{
	int l;            // 标签向量长度，当前为10
	float* LabelData; // 10维one-hot，正确类别为1，其余为0；不是84维RBF模板
}MinstLabel;

typedef struct MinstLabelArr{
	int LabelNum; // 标签样本数
	MinstLabel* LabelPtr; // 含LabelNum个标签结构体的数组
} MinstLabelArr, *LabelArr;              // LabelArr是MinstLabelArr的指针别名

LabelArr read_Lable(const char* filename); // 读取IDX标签字节并构造one-hot向量

// 读取28×28 IDX图片，返回四周填充2像素后的32×32数组，数值范围[-0.1, 1.175]。
ImgArr read_Img(const char* filename);

void save_Img(ImgArr imgarr,const char* filedir); // 历史工具：逐图写本机float数据，无文件头，非IDX/PNG/784字节raw

#endif
