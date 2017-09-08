CC = clang

INCLUDES = -I./headers

CXXFLAGS = -Wall

TARGET = dnn
SRC = $(wildcard *.cpp)
OBJECTS = $(SRC:.cpp=.o)
HEADERS = $(wildcard *.h)

$(TARGET): $(OBJECTS)

$(OBJECTS): $(HEADERS)

.PHONY: clean
clean:
	rm -f *~ *.o core $(TARGET)

.PHONY: all
all: clean lib