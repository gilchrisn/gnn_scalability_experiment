# Compiler
CXX = g++

# Compiler flags
# -O3 for optimization, -std=c++17 for modern C++
CXXFLAGS = -std=c++17 -O3 -Wall

# Source files
SRCS = HUB/main.cpp HUB/param.cpp

# Executable name
TARGET = bin/graph_prep

# Build rule
all: $(TARGET)

$(TARGET): $(SRCS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean rule
clean:
	rm -f $(TARGET)