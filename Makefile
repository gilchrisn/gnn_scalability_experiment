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
	@cmd /C "IF NOT EXIST bin mkdir bin"
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean rule
clean:
	@cmd /C "DEL /F /Q $(subst /,\,$(TARGET))"


# g++ -std=c++17 -O3 -o bin/graph_prep.exe HUB/main.cpp HUB/param.cpp