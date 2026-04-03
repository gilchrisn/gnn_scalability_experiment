/*********** Code for BoolAPCore^{G} ***********/

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>
#include <cstring>
#include <set>
#include <vector>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <unistd.h> 
using namespace std;
int NUM_THREADS = 1;
/*path of input files*/
char* HIN_PATH = (char*)"Movielens/graph_movielens.txt";
char* METAPATH_PATH = (char*)"Movielens/metapath.txt";

/*data structures*/
typedef struct {
	int v;
	int type;
}edge;
typedef vector<edge> vertice;

typedef struct {
	int n;
	int m;
	int t;
}graph;
graph g;

typedef vector<int> neighbor;

typedef struct {
	int i;
	int j;
}Tuple;
typedef vector<Tuple> tuples;

typedef int Index;

typedef struct vector<int> row_elements;

typedef struct {
	int row, col;
	vector<int> nnz_rows, nnz_cols; //record the number of each nnz row/line
	vector<Index> lhs_index, rhs_index; //record the index of each row/line (nnz: index in lhs/rhs_elements, else: -1)
	vector<row_elements> lhs_elements, rhs_elements;
}matrix;

vector<matrix> A;
vector<tuples> edgeTuples;
vector<int> path;
vector<int> core;
int path_length;
double BoolAPCoreG_runtime;

void read_env();
int load_HIN(char* filename);
int load_metapath(char* filename);
void BoolAPCoreG();
void parallel_core_decomposition(matrix Adjacent);
void initializeDegree(matrix Adjacent, int* degree, vector<int>& V, vector<neighbor> &neighbors);

/*matrix related functions*/
void setFromTuples(matrix &A, tuples t);
matrix transpose1(matrix A); //from rhs to lhs
matrix transpose2(matrix A); //from lhs to lhs
matrix sparseMatrixProduct(matrix A, matrix B);
int nonZeros(matrix A);


int main(int argc, char* argv[])
{
	HIN_PATH = argv[1];
	METAPATH_PATH = argv[2];

	read_env();
	load_HIN(HIN_PATH);
	load_metapath(METAPATH_PATH); 
	BoolAPCoreG();
	
	return 0;
}

void read_env()
{
#pragma omp parallel
	{
#pragma omp master
		NUM_THREADS = omp_get_num_threads();
	}
	printf("NUM_PROCS:     %d \n", omp_get_num_procs());
	printf("NUM_THREADS:   %d \n", NUM_THREADS);
}

int load_HIN(char* filename)
{
	FILE* infp = fopen(filename, "r");
	if (infp == NULL) {
		fprintf(stderr, "Error: could not open inputh file: %s.\n Exiting ...\n", filename);
		exit(1);
	}
	fprintf(stdout, "Reading HIN from: %s\n", filename);
	fscanf(infp, "%d %d %d\n", &(g.n), &(g.m), &(g.t));
	vertice tmp_v; tuples tmp_t; matrix tmp_m;
	tmp_m.row = tmp_m.col = g.n;
	for (int i = 0; i < g.n; i++) { edgeTuples.push_back(tmp_t); core.push_back(0);}
	for (int i = 0; i < g.t; i++) A.push_back(tmp_m);
	int u, v, type;
	edge e;
	while (fscanf(infp, "%d %d %d\n", &u, &v, &type) != EOF) {
		Tuple t = {u, v};
		edgeTuples[type - 1].push_back(t);
	}
	fclose(infp);
	for (int i = 0; i < g.t; i++)
	{
		setFromTuples(A[i], edgeTuples[i]);
	}
	return 0;
}

int load_metapath(char* filename)
{
	FILE* infp = fopen(filename, "r");
	if (infp == NULL) {
		fprintf(stderr, "Error: could not open inputh file: %s.\n Exiting ...\n", filename);
		exit(1);
	}
	fprintf(stdout, "Reading metapath: %s\n\n", filename);
	fscanf(infp, "%d\n", &path_length);
	int type;
	for (int i = 1; i < path_length; i++)
	{
		fscanf(infp, "%d ", &type);
		path.push_back(type);
	}
	fscanf(infp, "%d", &type);
	path.push_back(type);
	return 0;
}

void BoolAPCoreG()
{
	double run_time;
	timeval time_start;
	timeval time_over;
	timeval time1, time2;
	cout << "Ready to run BoolAPCore(G)..." << endl;
	gettimeofday(&time_start,NULL);
	gettimeofday(&time1,NULL);
	matrix B = { g.n, g.n };
	int target_num = 0;
	if (path[0] > 0) B = A[path[0] - 1];
	else B = transpose1(A[abs(path[0]) - 1]);
	double mean_density = 0;
	for (long long i = 1; i < path_length / 2; i++)
	{
		mean_density += double(nonZeros(B)) / g.n / g.n;
		if (path[i] > 0) B = sparseMatrixProduct(B, A[path[i] - 1]);
		else B = sparseMatrixProduct(B, transpose1(A[abs(path[i]) - 1]));
	}
	if(path_length>2) 
	{
		cout << "mean density=" << mean_density / (path_length / 2 - 1) << endl;
		double cur_density = double(nonZeros(B)) / g.n / g.n;
		cout << "intermidiate density=" << cur_density << endl;
		double decision = 1.55584046 + 0.0695962647 * (mean_density / cur_density) + 241.063437 * mean_density -508.697917 * cur_density;
				
		if (decision < 1)
		{
			cout << "    *The matrix is not appropriate for transpose" << endl;
			for (long long i = path_length / 2; i < path_length; i++)
			{
				if (path[i] > 0) B = sparseMatrixProduct(B, A[path[i] - 1]);
				else B = sparseMatrixProduct(B, transpose1(A[abs(path[i]) - 1]));
			}
		}
		else B = sparseMatrixProduct(B, transpose2(B));
	}
	else
	{
	    if(path[0] > 0) B = sparseMatrixProduct(B, transpose1(B));
	    else B = sparseMatrixProduct(B, A[abs(path[0]) - 1]);
	}

	gettimeofday(&time2,NULL);
	run_time = ((time2.tv_usec-time1.tv_usec)+(time2.tv_sec-time1.tv_sec)*1000000); //us
	cout << "running time for building Gp: " << run_time / 1000 << "ms\n";
	cout<<"Edges in Gp: "<<nonZeros(B)<<endl;

	gettimeofday(&time1,NULL);
	parallel_core_decomposition(B);
	gettimeofday(&time2,NULL);
	run_time = ((time2.tv_usec-time1.tv_usec)+(time2.tv_sec-time1.tv_sec)*1000000); //us
	cout << "running time for core decomposition: " << run_time / 1000 << "ms\n";

	gettimeofday(&time_over,NULL);
	run_time = ((time_over.tv_usec-time_start.tv_usec)+(time_over.tv_sec-time_start.tv_sec)*1000000); //us
	cout << "running time of BoolAPcore(G):" << run_time / 1000 << "ms\n\n";
	BoolAPCoreG_runtime = run_time;
}

void parallel_core_decomposition(matrix Adjacent)
{
	omp_lock_t lock;
	omp_init_lock(&lock);
	int* degree = new int[g.n + 1];
	vector<int> V;
	vector<neighbor> neighbors;
	initializeDegree(Adjacent, degree, V, neighbors);
	bool flg = 1;
	while (flg)
	{
		flg = 0;
#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < V.size(); i++)
		{
			int new_degree = 0;
			vector<int> cur_neighbors = neighbors[V[i]];
			if (cur_neighbors.size() == 0)
			{
				new_degree = 0;
			}
			else
			{
				int h_index = 0;
				int max_h = cur_neighbors.size();
				int* sum = new int[max_h + 1];
				int tot = 0;
				for (int k = 1; k <= max_h; k++) sum[k] = 0;

				for (int k = 0; k < max_h; k++)
				{
					if (degree[cur_neighbors[k]] >= max_h) sum[max_h]++;
					else sum[degree[cur_neighbors[k]]]++;
				}

				for (int k = max_h; k > 0; k--)
				{
					tot += sum[k];
					if (tot >= k)
					{
						new_degree = k;
						break;
					}
				}
			}

			if (new_degree != degree[V[i]])
			{
				degree[V[i]] = new_degree;
				omp_set_lock(&lock);
				flg = 1;
				omp_unset_lock(&lock);
			}
		}
#pragma omp barrier
	}
	omp_destroy_lock(&lock);
#pragma omp parallel for num_threads(NUM_THREADS)
	for(int i = 0; i < g.n; i++) core[i] = degree[i];
#pragma omp barrier
}

void initializeDegree(matrix Adjacent, int* degree, vector<int>& V, vector<neighbor>& neighbors)
{
	vector<int> empty_neighbors;
	for (int r = 0; r < g.n; r++) {neighbors.push_back(empty_neighbors); degree[r] = 0;}
	V = Adjacent.nnz_rows;
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int r = 0; r < V.size(); r++)
	{
		int row_num = V[r];
		int row_index = Adjacent.lhs_index[row_num];
		for (auto element: Adjacent.lhs_elements[row_index])
		{
			if (element != row_num) neighbors[row_num].push_back(element);
		}
		degree[row_num] = neighbors[row_num].size();
	}
//#pragma omp barrier
}

/****************************************************************************************************************************/
/************************************************  Matrix Related Functions  ************************************************/
/****************************************************************************************************************************/

bool cmp1(Tuple a, Tuple b)
{
	if (a.i == b.i) return a.j < b.j;
	else return a.i < b.i;
}

bool cmp2(Tuple a, Tuple b)
{
	if (a.j == b.j) return a.i < b.i;
	else return a.j < b.j;
}

void setFromTuples(matrix& A, tuples t)
{
	if(t.size()==0)
	{
		vector<int> empty_nnz_rows;
		A.nnz_rows = A.nnz_cols = empty_nnz_rows;
		return;
	}

	for(int r = 0; r < A.row; r++) A.lhs_index.push_back(-1);
	sort(t.begin(), t.end(), cmp1);
	int row = t[0].i;
	row_elements empty_elements;
	A.nnz_rows.push_back(row); 
	A.lhs_elements.push_back(empty_elements);
	Index idx = 0;
	A.lhs_index[row] = idx;
	for (int element = 0; element < t.size(); element++)
	{
		if (t[element].i > row)
		{
			idx += 1;
			A.lhs_elements.push_back(empty_elements);
			row = t[element].i;
			A.lhs_index[row] = idx;
			A.nnz_rows.push_back(row);
		}
		if(A.lhs_elements[idx].size() == 0 || A.lhs_elements[idx][A.lhs_elements[idx].size()-1] != t[element].j)
			A.lhs_elements[idx].push_back(t[element].j);
	}

	sort(t.begin(), t.end(), cmp2);
	int col = t[0].j;
	idx = 0;
	for (int c = 0; c < A.col; c++) A.rhs_index.push_back(-1);
	A.nnz_cols.push_back(col);
	A.rhs_elements.push_back(empty_elements);
	A.rhs_index[col] = idx;
	for (int element = 0; element < t.size(); element++)
	{
		if (t[element].j > col)
		{
			idx += 1;
			A.rhs_elements.push_back(empty_elements);
			col = t[element].j;
			A.rhs_index[col] = idx;
			A.nnz_cols.push_back(col);
		}
		if(A.rhs_elements[idx].size() == 0 || A.rhs_elements[idx][A.rhs_elements[idx].size()-1] != t[element].i)
			A.rhs_elements[idx].push_back(t[element].i);
	}
}

matrix transpose1(matrix A) //from rhs to lhs
{
	matrix B;
	B.col = A.row;
	B.row = A.col;
	B.nnz_rows = A.nnz_cols;
	B.lhs_index = A.rhs_index;
	B.lhs_elements = A.rhs_elements;
	return B;
}

matrix transpose2(matrix A) //from lhs to lhs
{
	matrix B;
	tuples t;
	B.col = A.row;
	B.row = A.col;
	B.lhs_index.assign(B.row, -1);
	for (int r = 0; r < A.nnz_rows.size(); r++)
	{
		Index row_index = A.lhs_index[A.nnz_rows[r]];
		for(auto element: A.lhs_elements[row_index])
		{
			t.push_back({element, r});
		}
	}

	if(t.size()==0)
	{
		vector<int> empty_nnz_rows;
		B.nnz_rows = B.nnz_cols = empty_nnz_rows;
		return B;
	}

	sort(t.begin(), t.end(), cmp1);
	int row = t[0].i;
	row_elements empty_elements;
	B.nnz_rows.push_back(row); 
	B.lhs_elements.push_back(empty_elements);
	Index idx = 0;
	B.lhs_index[row] = idx;
	for (int element = 0; element < t.size(); element++)
	{
		if (t[element].i > row)
		{
			idx += 1;
			B.lhs_elements.push_back(empty_elements);
			row = t[element].i;
			B.lhs_index[row] = idx;
			B.nnz_rows.push_back(row);
		}
		B.lhs_elements[idx].push_back(t[element].j);
	}
	return B;
}

matrix sparseMatrixProduct(matrix A, matrix B) //only need lhs format
{
	matrix C;
	C.row = A.row;
	C.col = B.col;
	row_elements empty_elements;
	int last_size = 0, cur_size;
	int max_nnz_rows = A.nnz_rows.size();
	for (int cur_row = 0; cur_row < max_nnz_rows; cur_row++)
	{
		C.lhs_elements.push_back(empty_elements);
	}
	C.lhs_index.assign(C.row, -1);
	vector<vector<int>> thread_sum(NUM_THREADS, vector<int>(C.col, 0));
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int cur_row = 0; cur_row < max_nnz_rows; cur_row++)
	{
		int tid = omp_get_thread_num();
		vector<int>& sum = thread_sum[tid];
		int row_num = A.nnz_rows[cur_row];
		int row_index = A.lhs_index[row_num];
		set<int> cur_elements;
		
		row_elements row_a_elements = A.lhs_elements[row_index];
		for (auto j: row_a_elements)
		{
			if (B.lhs_index[j] == -1) continue;
			else
			{
				for (auto k: B.lhs_elements[B.lhs_index[j]])
				{
					if (sum[k] == 0)
					{
						sum[k] = 1;
						C.lhs_elements[cur_row].push_back(k);
					}
				}
			}
		}
		if (C.lhs_elements[cur_row].size() == 0) continue;
		else
		{
			C.lhs_index[row_num] = cur_row;
			for (auto e: C.lhs_elements[cur_row]) sum[e] = 0;
		}
	}
	#pragma omp barrier
	for (int cur_row = 0; cur_row < max_nnz_rows; cur_row++)
	{
		if (C.lhs_elements[cur_row].size()>0) C.nnz_rows.push_back(A.nnz_rows[cur_row]);
	}
	return C;
}

int nonZeros(matrix A)
{
	int nnz_num = 0;
	for (int i = 0; i < A.lhs_elements.size(); i++) nnz_num += A.lhs_elements[i].size();
	return nnz_num;
}
