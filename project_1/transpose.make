
CC = g++
NVCC = nvcc
CFLAGS = -c -Wall -std=c++11 -Iinclude
CU_CFLAGS = -c -pg -std=c++11 -Iinclude -Xcompiler -Wall
LDFLAGS = -lboost_program_options
SOURCES = transpose_main.cpp parameters.cpp shared_utilities.cpp
# vector_add.cpp
CU_SOURCES = device_queries.cu transpose.cu
CU_OBJECTS = $(CU_SOURCES:.cu=.cu.o)
OBJECTS = $(SOURCES:.cpp=.cpp.o)
EXECUTABLE = transpose

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
