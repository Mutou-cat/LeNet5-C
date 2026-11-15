# DeepLearningC

使用标准 C11 手写的 MNIST 卷积神经网络，参考 LeNet-5，无深度学习框架依赖。
当前正在分模块改造为原版 LeNet-5：已完成32×32输入、缩放tanh、可训练下采样；
C3仍为12通道全连接卷积，尚未加入C5、F6和RBF输出，不是完整原版。

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
./build/cnn train CNN/Minst/train-images-idx3-ubyte CNN/Minst/train-labels-idx1-ubyte build/trained.cnn
./build/cnn test CNN/Minst/t10k-images-idx3-ubyte CNN/Minst/t10k-labels-idx1-ubyte build/trained.cnn
```

测试输出准确率和错误率。训练默认使用全部输入样本、10 个 epoch、学习率 1.0，
并将模型写入最后一个参数指定的位置；该位置已有文件会被覆盖。
模型采用 `DLCPOOL1` 格式：8字节标识、12个原生int结构字段，随后保存C1、C3、O5权重与偏置，
以及S2、S4的缩放系数与偏置；float及整数使用本机格式，尚非跨平台交换协议。
加载时检查版本、结构和数据完整性。旧的 `minst.cnn`、`my_model.cnn` 不适用于新网络，
需要重新训练；原文件未被删除或覆盖。`make clean` 会删除build目录，包括其中的模型。
学习率1.0是遗留默认值，尚未针对新激活和可训练下采样调参，完整数据集准确率未验证。

## 手写识别桌面窗口

在项目根目录运行：

```sh
make gui
```

窗口使用 Python 3 自带的 Tkinter（当前系统已可用），不依赖 Pillow、NumPy 或深度学习框架。
也可以先 `make`，再执行 `python3 gui.py`。若指定 Python 解释器，使用
`make gui PYTHON=/path/to/python3`。

1. 点击「选择」加载本分支重新训练生成的模型（例如 `build/trained.cnn`）。旧默认模型会被拒绝。
2. 在 28×28 像素网格上用鼠标左键书写，右键擦除。每格对应一个像素，可多画几笔加粗。
3. 点击「识别」或按回车，显示预测数字、实际输入预览和 0–9 的输出分数。
4. 点击「清空重写」开始下一个数字；也可以补画后再次识别。

画板直接存储 28×28 个像素（黑色 0、白色 255），不裁剪、不缩放、不自动居中。
每格在屏幕上显示为 10×10，方便鼠标操作；网格线及田字中线仅用于辅助书写，不进入模型。
GUI 将这 784 个灰度字节通过标准输入传给 `build/cnn predict MODEL`，
实际前向计算仍使用 GCC 编译的 C 代码，每次识别在独立子进程中完成。
C端与训练一致：四周填充2像素到32×32，归一化到[-0.1, 1.175]；窗口预览显示的是填边前的28×28网格。
显示的是当前输出层的缩放tanh输出，不是经过校准的概率，也不代表识别准确率。
模型只学习了 0–9，非数字图形也可能被预测为某个数字；鼠标手写效果可能低于 MNIST 测试集准确率。

## 源码

- `gui.py`：Tkinter 桌面画板、图像预处理和结果展示。
- `CNN/CNN/main.c`：训练、测试和单图识别命令行入口。
- `CNN/CNN/cnn.c`、`cnn.h`：网络、训练、推理和模型读写。
- `CNN/CNN/mat.c`、`mat.h`：矩阵、卷积和采样运算。
- `CNN/CNN/minst.c`、`minst.h`：MNIST IDX 数据读取。
- `CNN/CNN/io.h`：检查二进制输入是否完整。

当前结构（高×宽×通道）：32×32×1 → C1 28×28×6 → S2 14×14×6 →
C3 10×10×12 → S4 5×5×12 → 展平300维 → O5 10类。

## 可训练下采样 S2 / S4

每通道共享系数a、偏置b：`sum=窗口元素之和`，`z=a*sum+b`，
`y=1.7159*tanh((2/3)*z)`。2×2窗口的a初始化为0.25，b初始化为0。
`poolff`缓存窗口和；`poolbp`把dL/dy转换为dL/dz，并计算
`dscaleData=sum(d*窗口和)`、`dbasicData=sum(d)`。
传给窗口内各输入的梯度为`a*d`，再乘前一个卷积层的激活导数；不再固定除以4。
`cnnapplygrads`在反传完成后统一更新包括下采样在内的参数。

运行检查：

```sh
make test
```

检查包含下采样前向、系数/偏置/输入的有限差分、S2/S4网络集成、参数更新、模型保存重载、
不兼容/损坏模型拒绝，以及合成数据训练与predict预处理一致性。
本次未改动此前发现的O5偏置重复相加问题（其前向为Wx+2b，但偏置更新仅使用d），
输出层后续改造成RBF时需处理；这些测试不声称验证该遗留偏置梯度。

旧的 `cnn - 副本` 是与现有接口不兼容、未参与原项目构建的历史草稿，已移除；
旧 VS 工程、缓存及 Windows 编译产物已清理，当前构建使用根目录 Makefile。
代码与中文注释采用 UTF-8。

原项目教程：http://blog.csdn.net/tostq/article/details/51786265
