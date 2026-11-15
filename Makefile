CC = gcc
CPPFLAGS ?=
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS = -lm

SOURCES := CNN/CNN/main.c CNN/CNN/cnn.c CNN/CNN/mat.c CNN/CNN/minst.c
OBJECTS := $(patsubst CNN/CNN/%.c,build/%.o,$(SOURCES))
TARGET := build/cnn

.PHONY: all clean gui test
all: $(TARGET)

PYTHON ?= python3
gui: $(TARGET)
	$(PYTHON) gui.py

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/%.o: CNN/CNN/%.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

build:
	mkdir -p $@

clean:
	rm -rf build

-include $(OBJECTS:.o=.d)

TEST_OBJECTS := build/cnn.o build/mat.o build/minst.o
build/test_pool: tests/test_pool.c $(TEST_OBJECTS) CNN/CNN/cnn.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -ICNN/CNN $< $(TEST_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

build/test_sparse: tests/test_sparse.c $(TEST_OBJECTS) CNN/CNN/cnn.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -ICNN/CNN $< $(TEST_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

build/test_output: tests/test_output.c $(TEST_OBJECTS) CNN/CNN/cnn.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -ICNN/CNN $< $(TEST_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

build/test_c5: tests/test_c5.c $(TEST_OBJECTS) CNN/CNN/cnn.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -ICNN/CNN $< $(TEST_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TARGET) build/test_pool build/test_sparse build/test_output build/test_c5
	./build/test_c5
	./build/test_output
	./build/test_sparse
	$(PYTHON) tests/test_model.py
