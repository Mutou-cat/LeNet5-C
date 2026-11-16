/* 二维矩阵统一用mat[行][列]，nSize的初始化顺序是{列数, 行数}。
 * 返回float**的函数均新分配各行和行指针数组，不修改传入矩阵；调用方须逐行free后free外层。
 * addmat/multifactor写入调用方已分配的res，允许res与输入是同一矩阵。 */
#ifndef __MAT_
#define __MAT_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define full 0 // 每维输出长度=N+K-1
#define same 1 // 当前仅奇数核保持输入尺寸；双偶数核会每维少1
#define valid 2 // 同奇偶核的每维输出长度=N-K+1，要求输入不小于核

typedef struct Mat2DSize{
	int c; // 列（宽）
	int r; // 行（高）
}nSize;

float** rotate180(float** mat, nSize matSize);// 矩阵翻转180度

void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);// 矩阵相加

float** correlation(float** map,nSize mapSize,float** inputData,nSize inSize,int type);// 互相关

// 数学卷积：先旋转map再互相关；不能仅在前向删除翻转而保持反传公式不变。
float** cov(float** map,nSize mapSize,float** inputData,nSize inSize,int type);

// 按列复制upc次、按行复制upr次；仅复制数值，不乘池化系数或激活导数。
float** UpSample(float** mat,nSize matSize,int upc,int upr);

// 左右各补addc列0，上下各补addr行0。
float** matEdgeExpand(float** mat,nSize matSize,int addc,int addr);

// 左右各裁shrinkc列，上下各裁shrinkr行，调用方须保证结果尺寸为正。
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr);

void savemat(float** mat,nSize matSize,const char* filename);// 按行写本机float，无尺寸头或字节序转换

void multifactor(float** res, float** mat, nSize matSize, float factor);// 矩阵乘以系数

float summat(float** mat,nSize matSize);// 矩阵各元素的和

// 以下字符串工具实现在minst.c；返回堆内存，调用方负责free。
char * combine_strings(const char *a, const char *b);

// 将非负整数转为十进制字符串。
char* intTochar(int i);

#endif
