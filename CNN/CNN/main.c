#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn.h"
#include "minst.h"

static void usage(const char *program)
{
    printf("Usage: %s test|train IMAGES LABELS MODEL\n", program);
    printf("       %s predict MODEL < pixels.raw (784 grayscale bytes)\n", program);
    printf("MNIST images must be 28x28 IDX files; labels must be IDX files.\n");
    printf("Training uses 10 epochs and learning rate 0.001.\n");
}

static int predict(const char *model)
{
    unsigned char pixels[28 * 28];
    if (fread(pixels, 1, sizeof(pixels), stdin) != sizeof(pixels) ||
        fgetc(stdin) != EOF) {
        fprintf(stderr, "Expected exactly 784 grayscale bytes on stdin\n");
        return EXIT_FAILURE;
    }
    /* Keep the raw drawing 28x28; use exactly the same preprocessing as read_Img. */
    float data[32][32];
    float *rows[32];
    for (int r = 0; r < 32; ++r) {
        rows[r] = data[r];
        for (int c = 0; c < 32; ++c) data[r][c] = -0.1f;
    }
    for (int r = 0; r < 28; ++r)
        for (int c = 0; c < 28; ++c)
            data[r+2][c+2] = -0.1f + 1.275f * (pixels[r*28+c]/255.0f);
    CNN cnn = {0};
    nSize size = {32, 32};
    cnnsetup(&cnn, size, 10);
    importcnn(&cnn, model);
    cnnff(&cnn, rows);
    int best = 0;
    for (int i = 0; i < 10; ++i) {
        if (!isfinite(cnn.O7.y[i])) {
            fprintf(stderr, "Model produced a non-finite RBF distance\n");
            return EXIT_FAILURE;
        }
        if (cnn.O7.y[i] < cnn.O7.y[best]) best = i;
    }
    printf("{\"digit\":%d,\"score_type\":\"squared_distance\",\"scores\":[", best);
    for (int i = 0; i < 10; ++i)
        printf("%s%.8f", i ? "," : "", cnn.O7.y[i]);
    printf("]}\n");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "predict") == 0)
        return predict(argv[2]);
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 5 || (strcmp(argv[1], "test") != 0 &&
                      strcmp(argv[1], "train") != 0)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    ImgArr images = read_Img(argv[2]);
    LabelArr labels = read_Lable(argv[3]);
    if (images->ImgNum <= 0 || images->ImgNum != labels->LabelNum ||
        images->ImgPtr[0].c != 32 || images->ImgPtr[0].r != 32) {
        fprintf(stderr, "Expected matching, nonempty MNIST data with 32x32   images\n");
        return EXIT_FAILURE;
    }

    CNN cnn = {0};
    nSize input_size = {32, 32};
    cnnsetup(&cnn, input_size, 10);
    if (strcmp(argv[1], "train") == 0) {
        CNNOpts options = {10, 0.001f};
        cnntrain(&cnn, images, labels, options, images->ImgNum);
        savecnn(&cnn, argv[4]);
        printf("Training finished: %d samples; model saved to %s\n",
               images->ImgNum, argv[4]);
    } else {
        importcnn(&cnn, argv[4]);
        float error = cnntest(&cnn, images, labels, images->ImgNum);
        printf("Samples: %d, accuracy: %.2f%%, error rate: %.2f%%\n",
               images->ImgNum, (1.0f - error) * 100.0f, error * 100.0f);
    }
    return EXIT_SUCCESS;
}
