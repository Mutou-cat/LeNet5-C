#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "mat.h"

float** rotate180(float** mat, nSize matSize)// 矩阵翻转180度
{
	int i,c,r;
	int outSizeW=matSize.c;
	int outSizeH=matSize.r;
	float** outputData=(float**)malloc(outSizeH*sizeof(float*));
	for(i=0;i<outSizeH;i++)
		outputData[i]=(float*)malloc(outSizeW*sizeof(float));

	for(r=0;r<outSizeH;r++)
		for(c=0;c<outSizeW;c++)
			outputData[r][c]=mat[outSizeH-r-1][outSizeW-c-1];

	return outputData;
}

/* 先补零计算full互相关，再按type裁剪。map不翻转，cov负责数学卷积的翻转。
 * full: N+K-1；valid: N-K+1（本项目使用宽高同奇偶的核）。
 * same用对称裁剪：奇数核得到N，双偶数核得到N-1，并非一般库的非对称same规则。
 * 当前宽高奇偶不同的核没有独立处理；本网络用正方形核和full/valid路径。
 */

float** correlation(float** map,nSize mapSize,float** inputData,nSize inSize,int type)// 互相关
{
	// 正向cov调用这里，反向传播也直接使用；以下半尺寸用于确定裁剪边宽。
	int i,j,c,r;
	int halfmapsizew;
	int halfmapsizeh;
	if(mapSize.r%2==0&&mapSize.c%2==0){ // 核宽高均为偶数
		halfmapsizew=(mapSize.c)/2; // 用于后续裁剪的半宽
		halfmapsizeh=(mapSize.r)/2;
	}else{
		halfmapsizew=(mapSize.c-1)/2; // 用于后续裁剪的半宽
		halfmapsizeh=(mapSize.r-1)/2;
	}

	// full模式每维长度为输入长度+核长度-1。
	int outSizeW=inSize.c+(mapSize.c-1); // full输出宽度
	int outSizeH=inSize.r+(mapSize.r-1);
	float** outputData=(float**)malloc(outSizeH*sizeof(float*)); // 分配输出行指针，各行再单独分配
	for(i=0;i<outSizeH;i++)
		outputData[i]=(float*)calloc(outSizeW,sizeof(float));

	// 左右各补mapSize.c-1列0，上下各补mapSize.r-1行0。
	float** exInputData=matEdgeExpand(inputData,inSize,mapSize.c-1,mapSize.r-1);

	// full输出(j,i)的核左上角在原输入(j-(Kh-1), i-(Kw-1))，补零处理越界。
	for(j=0;j<outSizeH;j++)
		for(i=0;i<outSizeW;i++)
			for(r=0;r<mapSize.r;r++)
				for(c=0;c<mapSize.c;c++){
					outputData[j][i]=outputData[j][i]+map[r][c]*exInputData[j+r][i+c];
				}

	for(i=0;i<inSize.r+2*(mapSize.r-1);i++)
		free(exInputData[i]);
	free(exInputData);

	nSize outSize={outSizeW,outSizeH};
	switch(type){ // 裁剪会产生新矩阵，返回前释放不再使用的full结果
	case full: // 直接交给调用方释放
		return outputData;
	case same:{
		float** sameres=matEdgeShrink(outputData,outSize,halfmapsizew,halfmapsizeh);
		for(i=0;i<outSize.r;i++)
			free(outputData[i]);
		free(outputData);
		return sameres;
		}
	case valid:{
		// 宽高同奇偶时，每侧裁K-1个位置，只保留核完全落在输入内的结果。
		float** validres;
		if(mapSize.r%2==0&&mapSize.c%2==0)
			validres=matEdgeShrink(outputData,outSize,halfmapsizew*2-1,halfmapsizeh*2-1);
		else
			validres=matEdgeShrink(outputData,outSize,halfmapsizew*2,halfmapsizeh*2);
		for(i=0;i<outSize.r;i++)
			free(outputData[i]);
		free(outputData);
		return validres;
		}
	default: // 历史行为：未知type按full返回，没有参数报错。
		return outputData;
	}
}

float** cov(float** map,nSize mapSize,float** inputData,nSize inSize,int type) // 卷积操作
{
	// 卷积操作可以用旋转180度的特征模板相关来求
	float** flipmap=rotate180(map,mapSize); //旋转180度的特征模板
	float** res=correlation(flipmap,mapSize,inputData,inSize,type);
	int i;
	for(i=0;i<mapSize.r;i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}

// 每个元素复制为upr行×upc列的常值块；池化反传中的乘系数由调用方完成。
float** UpSample(float** mat,nSize matSize,int upc,int upr)
{
	int i,j,m,n;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r*upr)*sizeof(float*)); // 结果的初始化
	for(i=0;i<(r*upr);i++)
		res[i]=(float*)malloc((c*upc)*sizeof(float));

	for(j=0;j<r*upr;j=j+upr){
		for(i=0;i<c*upc;i=i+upc)// 在输出块的首行横向复制
			for(m=0;m<upc;m++)
				res[j][i+m]=mat[j/upr][i/upc];

		for(n=1;n<upr;n++)      // 将该首行向下复制
			for(i=0;i<c*upc;i++)
				res[j+n][i]=res[j][i];
	}
	return res;
}

// 左右各补addc列0，上下各补addr行0，保留中心区域数据。
float** matEdgeExpand(float** mat,nSize matSize,int addc,int addr)
{
	int i,j;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r+2*addr)*sizeof(float*)); // 结果的初始化
	for(i=0;i<(r+2*addr);i++)
		res[i]=(float*)malloc((c+2*addc)*sizeof(float));

	for(j=0;j<r+2*addr;j++){
		for(i=0;i<c+2*addc;i++){
			if(j<addr||i<addc||j>=(r+addr)||i>=(c+addc))
				res[j][i]=(float)0.0;
			else
				res[j][i]=mat[j-addr][i-addc]; // 复制输入矩阵对应位置
		}
	}
	return res;
}

// 左右各裁shrinkc列，上下各裁shrinkr行；需保证裁剪后尺寸为正。
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr)
{
	int i,j;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r-2*shrinkr)*sizeof(float*)); // 结果矩阵的初始化
	for(i=0;i<(r-2*shrinkr);i++)
		res[i]=(float*)malloc((c-2*shrinkc)*sizeof(float));


	for(j=0;j<r;j++){
		for(i=0;i<c;i++){
			if(j>=shrinkr&&i>=shrinkc&&j<(r-shrinkr)&&i<(c-shrinkc))
				res[j-shrinkr][i-shrinkc]=mat[j][i]; // 复制输入矩阵对应位置
		}
	}
	return res;
}

// 历史导出工具：连续逐行写float，不保存行列数，也不是图片编码格式。
void savemat(float** mat,nSize matSize,const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL) {
		perror(filename);
		exit(EXIT_FAILURE);
	}

	int i;
	for(i=0;i<matSize.r;i++)
		fwrite(mat[i],sizeof(float),matSize.c,fp);
	fclose(fp);
}

// 调用方须保证两个输入同尺寸；当前尺寸不匹配只打印提示，不会停止循环。
void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)// 矩阵相加
{
	int i,j;
	if(matSize1.c!=matSize2.c||matSize1.r!=matSize2.r)
		printf("ERROR: Size is not same!");

	for(i=0;i<matSize1.r;i++)
		for(j=0;j<matSize1.c;j++)
			res[i][j]=mat1[i][j]+mat2[i][j];
}

void multifactor(float** res, float** mat, nSize matSize, float factor)// 矩阵乘以系数
{
	int i,j;
	for(i=0;i<matSize.r;i++)
		for(j=0;j<matSize.c;j++)
			res[i][j]=mat[i][j]*factor;
}

float summat(float** mat,nSize matSize) // 矩阵各元素的和
{
	float sum=0.0;
	int i,j;
	for(i=0;i<matSize.r;i++)
		for(j=0;j<matSize.c;j++)
			sum=sum+mat[i][j];
	return sum;
}
