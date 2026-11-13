#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn.h"
#include "minst.h"

static void usage(const char *program)
{
    printf("Usage: %s test|train IMAGES LABELS MODEL\n", program);
    printf("MNIST images must be 28x28 IDX files; labels must be IDX files.\n");
    printf("Training uses one epoch and learning rate 1.0.\n");
}

int main(int argc, char **argv)
{
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
        images->ImgPtr[0].c != 28 || images->ImgPtr[0].r != 28) {
        fprintf(stderr, "Expected matching, nonempty MNIST data with 28x28 images\n");
        return EXIT_FAILURE;
    }

    CNN cnn = {0};
    nSize input_size = {28, 28};
    cnnsetup(&cnn, input_size, 10);
    if (strcmp(argv[1], "train") == 0) {
        CNNOpts options = {1, 1.0f};
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
