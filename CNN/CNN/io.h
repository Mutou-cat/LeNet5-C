#ifndef CNN_IO_H
#define CNN_IO_H

#include <stdio.h>
#include <stdlib.h>

static inline void read_exact(void *buffer, size_t size, size_t count, FILE *fp)
{
    if (fread(buffer, size, count, fp) != count) {
        fprintf(stderr, "Invalid or truncated input file\n");
        exit(EXIT_FAILURE);
    }
}

static inline void write_exact(const void *buffer, size_t size, size_t count, FILE *fp)
{
    if (fwrite(buffer, size, count, fp) != count) {
        fprintf(stderr, "Failed to write model file\n");
        exit(EXIT_FAILURE);
    }
}

#endif
