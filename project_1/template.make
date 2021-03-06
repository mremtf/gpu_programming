
CC = g++
NVCC = nvcc
CFLAGS = -c -Wall -std=c++11 -Iinclude
CU_CFLAGS = -c -std=c++11 -Iinclude
LDFLAGS = -lboost_program_options
SOURCES = vector_main.cpp parameters.cpp
# vector_add.cpp
CU_SOURCES = device_queries.cu vector_add.cu
CU_OBJECTS = $(CU_SOURCES:.cu=.cu.o)
OBJECTS = $(SOURCES:.cpp=.cpp.o)
EXECUTABLE = vector_add

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(LDFLAGS) $(OBJECTS) $(CU_OBJECTS) -o $@

%.cpp.o: src/%.cpp
	$(CC) $(CFLAGS) $< -o $@

%.cu.o: src/%.cu
	$(NVCC) $(CU_CFLAGS) $< -o $@

clean:
	$(RM) *.o $(EXECUTABLE)

.PHONY: clean
