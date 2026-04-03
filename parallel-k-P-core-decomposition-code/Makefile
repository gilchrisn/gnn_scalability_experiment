# BoolAP Core Decomposition — Build Targets
#
# These programs use POSIX-only headers (sys/time.h, unistd.h).
# On Windows, compile via WSL (Windows Subsystem for Linux):
#
#   wsl make              # Build all three targets
#   wsl make BoolAPCoreD  # Build only BoolAPCoreD
#
# The compiled binaries are placed in this directory.
# BoolAPRunner in src/bridge/boolap.py handles WSL path conversion
# automatically when invoking from Windows Python.

CXX      := g++
CXXFLAGS := -fopenmp -O2 -std=c++11

.PHONY: all clean

all: BoolAPCoreD BoolAPCoreG HomBCore

BoolAPCoreD: BoolAPCoreD.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

BoolAPCoreG: BoolAPCoreG.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

HomBCore: HomBCore.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f BoolAPCoreD BoolAPCoreG HomBCore
