CC = clang

INCLUDES = -I./headers

CFLAGS = -Wall

TARGET = dnn
SRC = $(wildcard *.c)
OBJECTS = $(SRC:.c=.o)
HEADERS = $(wildcard *.h)

$(TARGET): $(OBJECTS)

$(OBJECTS): $(HEADERS)

.PHONY: clean
clean:
	rm -f *~ *.o core $(TARGET)

.PHONY: all
all: clean lib
