# 默认使用系统GCC；可用make CC=...在命令行覆盖。
CC = gcc
CPPFLAGS ?=
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS = -lm

# 仅编译运行时源码，不包含临时验证程序。
SOURCES := CNN/CNN/main.c CNN/CNN/cnn.c CNN/CNN/mat.c CNN/CNN/minst.c CNN/CNN/rbf.c
OBJECTS := $(patsubst CNN/CNN/%.c,build/%.o,$(SOURCES))
TARGET := build/cnn

.PHONY: all clean gui
all: $(TARGET)

PYTHON ?= python3
# 启动窗口前确保C推理程序已构建。
gui: $(TARGET)
	$(PYTHON) gui.py

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# -MMD/-MP生成头文件依赖，修改.h后会重新编译依赖它的.c。
build/%.o: CNN/CNN/%.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

build:
	mkdir -p $@

# 删除整个build目录；长期保留的模型和日志应放在其他位置。
clean:
	rm -rf build

-include $(OBJECTS:.o=.d)
