# DeepLearningC

使用标准 C11 手写的 MNIST 卷积神经网络，参考 LeNet-5，无深度学习框架依赖。
卷积、平均池化、Sigmoid、反向传播和参数更新均由 C 实现。

## 使用 GCC 构建

在项目根目录执行：

```sh
make
./build/cnn --help
```

需要 GCC、GNU Make 和系统数学库（链接参数 `-lm`）。Makefile 默认使用 `gcc`，
开启 `-std=c11 -Wall -Wextra -Wpedantic`；执行 `make clean` 清理构建目录。

## 测试与训练

自行准备并解压 MNIST 的 IDX 图片、标签文件，仓库不包含数据集。
参数中的路径按实际文件名填写（测试集通常使用 `t10k-` 前缀）：

```sh
./build/cnn test CNN/Minst/t10k-images-idx3-ubyte CNN/Minst/t10k-labels-idx1-ubyte CNN/CNN/minst.cnn
./build/cnn train CNN/Minst/train-images-idx3-ubyte CNN/Minst/train-labels-idx1-ubyte build/trained.cnn
```

测试输出准确率和错误率。训练默认使用全部输入样本、1 个 epoch、学习率 1.0，
并将模型写入最后一个参数指定的位置；该位置已有文件会被覆盖。
模型保持原项目的二进制 float 格式，文件不包含网络结构及版本信息。

## 源码

- `CNN/CNN/main.c`：命令行入口。
- `CNN/CNN/cnn.c`、`cnn.h`：网络、训练、推理和模型读写。
- `CNN/CNN/mat.c`、`mat.h`：矩阵、卷积和采样运算。
- `CNN/CNN/minst.c`、`minst.h`：MNIST IDX 数据读取。
- `CNN/CNN/io.h`：检查二进制输入是否完整。

输入 28×28×1 → 卷积 24×24×6 → 平均池化 12×12×6 →
卷积 8×8×12 → 平均池化 4×4×12 → 全连接 10 类。

旧的 `cnn - 副本` 是与现有接口不兼容、未参与原项目构建的历史草稿，已移除；
旧 VS 工程和已提交的 Windows 编译产物保留供参考，当前构建使用根目录 Makefile。
代码与中文注释采用 UTF-8。

原项目教程：http://blog.csdn.net/tostq/article/details/51786265
