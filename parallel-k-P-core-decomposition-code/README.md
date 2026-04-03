# Efficient Core Decomposition over Large Heterogeneous Information Networks

This is the official code release of the following paper:

Yucan Guo, Chenhao Ma, and Yixiang Fang. Efficient Core Decomposition over Large Heterogeneous Information Networks. ICDE 2024.

## Code files

* HomBCore.cpp: Including the source code of HomBCore.
* BoolAPCoreG.cpp: Including the source code of BoolAPCore<sup>G</sup> and DP-SpGEM-related functions.
* BoolAPCoreD.cpp: Including the source code of BoolAPCore<sup>D</sup> and DP-SpLDM-related functions.

## Compiling the program

```
$g++ -fopenmp -o HomBCore HomBCore.cpp
$g++ -fopenmp -o BoolAPCoreG BoolAPCoreG.cpp
$g++ -fopenmp -o BoolAPCoreD BoolAPCoreD.cpp
```

## Input format

* HIN file

  An HIN with vertex start from 0, and edge type start from 1. The first line of the file should contain 3 integers: the number of vertices, the number of edges, and the number of edge types.
* meta-path file

  - A meta-path consists of edge types, in which the negative numbers denote reverse edge types. The first line of the file is the length of the meta-path.
  - For HomBCore, the first line of the file should contain 2 integers: the length of the meta-path and the type of the first vertex in the meta-path.  
* vertices file

  - Record the vertex type of each vertex, in which the *i*-th line records the vertex type of the *i*-th vertex.
  - Necessary for HomBCore, other algorithms only need the HIN and meta-path files.

## Running the program
```
$./HomBCore [HIN file path] [vertices file path] [meta-path file path]
e.g. $./HomBCore Movielens/graph_Movielens.txt Movielens/vertices_Movielens.txt Movielens/metapath_HomBCore.txt

$./BoolAPCoreG [HIN file path] [meta-path file path]
e.g. $./BoolAPCoreG Movielens/graph_Movielens.txt Movielens/metapath.txt

$./BoolAPCoreD [HIN file path] [meta-path file path]
e.g. $./BoolAPCoreD Movielens/graph_Movielens.txt Movielens/metapath.txt
```
