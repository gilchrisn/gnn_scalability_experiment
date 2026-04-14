# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall

# Source files
SRCS = HUB/main.cpp HUB/param.cpp

# Executables
TARGET      = bin/graph_prep
MPRW_TARGET = bin/mprw_exec

# Build rule
all: $(TARGET) $(MPRW_TARGET)

$(TARGET): $(SRCS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

$(MPRW_TARGET): csrc/mprw_exec.cpp HUB/param.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $(MPRW_TARGET) csrc/mprw_exec.cpp HUB/param.cpp

# Clean rule
clean:
	rm -f $(TARGET) $(MPRW_TARGET)