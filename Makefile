CC = gcc
CPPFLAGS ?=
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS = -lm

SOURCES := CNN/CNN/main.c CNN/CNN/cnn.c CNN/CNN/mat.c CNN/CNN/minst.c
OBJECTS := $(patsubst CNN/CNN/%.c,build/%.o,$(SOURCES))
TARGET := build/cnn

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/%.o: CNN/CNN/%.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

build:
	mkdir -p $@

clean:
	rm -rf build

-include $(OBJECTS:.o=.d)
