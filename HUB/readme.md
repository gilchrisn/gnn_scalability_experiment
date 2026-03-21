# A Sketch Propagation Framework for Hub Queries on Unmaterialized Relational Graphs

Compile
-------
Before compile the program, set the key parameters for our program in file "param.h".

**There are 5 parameters:**
* K: the size of the KMV sketch
* L: the number of sketch propagations
* LOOP: take the average result of LOOP executions (for approximate HUB queries)
* QNUM: the number of personalized queries sampled for each candidate meta-path (for personalized HUB queries)
* PATHCOUNT: the number of meta-paths sampled for scalability test with lengh 1, 2, 3 respectively

Then, compile the code for HUB by executing the following command on linux:

```sh
g++ -O3 main.py -o HUB
```

Running code
-------

To run the code for HUB queries, execute the following command on linux:

```sh
./HUB dataset alg $\lambda$ $\beta$
```

**There are 5 parameters:**
* dataset: name of the KG
* alg: the algorithm to run
* $\lambda$: parameter for HUB problem
* $\beta$: parameter for early termination


For example, the following command execute the sketch propagation framework to find top-0.01 nodes (degree-based) in each meta-path graph induced by candidate meta-paths in imdb.

```sh
./HUB imdb GloD 0.01 0
```

Note that before running other algorithms, please first run the following four commands:

```sh
./HUB dataset ExactD $\lambda$ 0 > global_res/dataset/df1/hg_global_greater_r$\lambda$.res
./HUB dataset ExactD+ $\lambda$ 0 > global_res/dataset/df1/hg_global_r$\lambda$.res
./HUB dataset ExactH $\lambda$ 0 > global_res/dataset/hf1/hg_global_greater_r$\lambda$.res
./HUB dataset ExactH+ $\lambda$ 0 > global_res/dataset/hf1/hg_global_r$\lambda$.res
```

to store the result for exact methods, which is used for effectiveness evaluation.

Input Files
-----------
**The program HUB requires 4 input files:**
* node.dat stores nodes in KG.
* link.dat stores edges in KG.
* meta.dat stores the number of nodes in KG.
* dataset-cod-global-rules.dat stores the meta-paths mined by AnyBURL
