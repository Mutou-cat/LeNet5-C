#ifndef CNN_IO_H
#define CNN_IO_H

#include <stdio.h>
#include <stdlib.h>

// size是单个元素字节数，count是元素数；buffer至少容纳size*count字节。
// 按fread的size/count语义读取；短读或错误直接退出，数据含义由调用方校验。
static inline void read_exact(void *buffer, size_t size, size_t count, FILE *fp)
{
    if (fread(buffer, size, count, fp) != count) {
        fprintf(stderr, "Invalid or truncated input file\n");
        exit(EXIT_FAILURE);
    }
}

// 必须写满count个size字节元素；不转换字节序，不检查关闭文件时的刷新错误。
static inline void write_exact(const void *buffer, size_t size, size_t count, FILE *fp)
{
    if (fwrite(buffer, size, count, fp) != count) {
        fprintf(stderr, "Failed to write model file\n");
        exit(EXIT_FAILURE);
    }
}

#endif
