# Compiler
CXX = g++

# Compiler flags
# -O3 for optimization, -std=c++17 for modern C++
CXXFLAGS = -std=c++17 -O3 -Wall

# Source files
# --- THIS IS THE FIX ---
# Add HUB/param.cpp to the list of source files
SRCS = HUB/main.cpp HUB/param.cpp

# Executable name
TARGET = bin/graph_prep

# Build rule
all: $(TARGET)

# The 'mkdir' line is modified to be run *by* cmd.exe
$(TARGET): $(SRCS)
	@cmd /C "IF NOT EXIST bin mkdir bin"
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean rule
clean:
	@cmd /C "DEL /F /Q $(subst /,\,$(TARGET))"