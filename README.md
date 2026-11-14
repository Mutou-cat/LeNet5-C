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

测试输出准确率和错误率。训练默认使用全部输入样本、10 个 epoch、学习率 1.0，
并将模型写入最后一个参数指定的位置；该位置已有文件会被覆盖。
模型保持原项目的二进制 float 格式，文件不包含网络结构及版本信息。

## 手写识别桌面窗口

在项目根目录运行：

```sh
make gui
```

窗口使用 Python 3 自带的 Tkinter（当前系统已可用），不依赖 Pillow、NumPy 或深度学习框架。
也可以先 `make`，再执行 `python3 gui.py`。若指定 Python 解释器，使用
`make gui PYTHON=/path/to/python3`。

1. 默认选择 `CNN/CNN/my_model.cnn`，不存在时使用原始 `minst.cnn`；可点击「选择」切换模型。
2. 在 28×28 像素网格上用鼠标左键书写，右键擦除。每格对应一个像素，可多画几笔加粗。
3. 点击「识别」或按回车，显示预测数字、实际输入预览和 0–9 的输出分数。
4. 点击「清空重写」开始下一个数字；也可以补画后再次识别。

画板直接存储 28×28 个像素（黑色 0、白色 255），不裁剪、不缩放、不自动居中。
每格在屏幕上显示为 10×10，方便鼠标操作；网格线及田字中线仅用于辅助书写，不进入模型。
GUI 将这 784 个灰度字节通过标准输入传给 `build/cnn predict MODEL`，
实际前向计算仍使用 GCC 编译的 C 代码，每次识别在独立子进程中完成。
显示的是原模型的 Sigmoid 输出，不是经过校准的概率，也不代表识别准确率。
模型只学习了 0–9，非数字图形也可能被预测为某个数字；鼠标手写效果可能低于 MNIST 测试集准确率。

## 源码

- `gui.py`：Tkinter 桌面画板、图像预处理和结果展示。
- `CNN/CNN/main.c`：训练、测试和单图识别命令行入口。
- `CNN/CNN/cnn.c`、`cnn.h`：网络、训练、推理和模型读写。
- `CNN/CNN/mat.c`、`mat.h`：矩阵、卷积和采样运算。
- `CNN/CNN/minst.c`、`minst.h`：MNIST IDX 数据读取。
- `CNN/CNN/io.h`：检查二进制输入是否完整。

输入 28×28×1 → 卷积 24×24×6 → 平均池化 12×12×6 →
卷积 8×8×12 → 平均池化 4×4×12 → 全连接 10 类。

旧的 `cnn - 副本` 是与现有接口不兼容、未参与原项目构建的历史草稿，已移除；
旧 VS 工程、缓存及 Windows 编译产物已清理，当前构建使用根目录 Makefile。
代码与中文注释采用 UTF-8。

原项目教程：http://blog.csdn.net/tostq/article/details/51786265
