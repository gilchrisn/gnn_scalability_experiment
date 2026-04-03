/*********** Code for BoolAPCore^{D} ***********/

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <algorithm>
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

/*path of input files*/
char* HIN_PATH = (char*)"Movielens/graph_Movielens.txt";
char* METAPATH_PATH = (char*)"Movielens/metapath.txt";

int NUM_THREADS = 1;

/*data structures*/
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
typedef vector <Tuple> tuples;

typedef int Index;

typedef struct {
	int row, col;
	vector<int> nonzero_row, nonzero_col;
	vector<Index> lhs_index, rhs_index;
	vector<int> lhs_elements, rhs_elements;
}matrix;

vector<matrix> A;
vector<tuples> edgeTuples;
vector<int> path;
vector<int> core;
int path_length;
double BoolAPCoreD_runtime;

void read_env();
int load_HIN(char* filename);
int load_metapath(char* filename);
void BoolAPCoreD();
void parallel_core_decomposition(matrix Adjacent);
void initializeDegree(matrix Adjacent, int* degree, vector<int>& V, vector<neighbor> &neighbors);

/*Matrix related functions*/
void setFromTuples(matrix &A, tuples t);
matrix transpose(matrix A, int format);
matrix boolProduct(matrix A, matrix B);
matrix boolTransposeProduct(matrix A);
matrix non_parallel_boolProduct(matrix A, matrix B); 
int nonZeros(matrix A);


int main(int argc, char* argv[])
{
	HIN_PATH = argv[1];
	METAPATH_PATH = argv[2];

	read_env();
	load_HIN(HIN_PATH);
	load_metapath(METAPATH_PATH); 
	BoolAPCoreD();

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
	tuples tmp_t; matrix tmp_m;
	tmp_m.row = tmp_m.col = g.n;
	for (int i = 0; i < g.n; i++) {edgeTuples.push_back(tmp_t); core.push_back(0);}
	for (int i = 0; i < g.t; i++) A.push_back(tmp_m);
	int u, v, type;
	while (fscanf(infp, "%d %d %d\n", &u, &v, &type) != EOF) {
		Tuple t = {u, v};
		edgeTuples[type - 1].push_back(t);
	}
	fclose(infp);
	int memory_of_matrices = 0;
	for (int i = 0; i < g.t; i++)
	{
		setFromTuples(A[i], edgeTuples[i]);
		int len = 2 + A[i].nonzero_row.size() + A[i].nonzero_col.size() + A[i].lhs_index.size() + A[i].rhs_index.size() + A[i].lhs_elements.size() +A[i].rhs_elements.size();
		memory_of_matrices += sizeof(int) * len;
	}
	cout << "The memory occupied by the initial matrices is: " << memory_of_matrices/1024 << " kbytes" << endl;
	return 0;
}

int load_metapath(char* filename)
{
	FILE* infp = fopen(filename, "r");
	if (infp == NULL) {
		fprintf(stderr, "Error: could not open inputh file: %s.\n Exiting ...\n", filename);
		exit(1);
	}
	fprintf(stdout, "Reading metapath from: %s\n\n", filename);
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

void BoolAPCoreD()
{
	double run_time;
	timeval time_start;
	timeval time_over;
	timeval time1, time2;
	cout << "Ready to run BoolAPCore(D)..." << endl;
	gettimeofday(&time_start,NULL);
	gettimeofday(&time1,NULL);
	matrix B = { g.n, g.n };
	int target_num = 0;
	if (path[0] > 0) B = A[path[0] - 1];
	else B = transpose(A[abs(path[0]) - 1], 0);
	for (long long i = 1; i < path_length / 2; i++)
	{
		if (path[i] > 0) B = boolProduct(B, A[path[i] - 1]);
		else B = boolProduct(B, transpose(A[abs(path[i]) - 1], 1));
	}
	double cur_density = double(nonZeros(B)) / g.n / g.n;
	if(NUM_THREADS > 1) B = boolProduct(B, transpose(B, 1));
	else B = boolTransposeProduct(B);
	gettimeofday(&time2,NULL);
	run_time= ((time2.tv_usec-time1.tv_usec)+(time2.tv_sec-time1.tv_sec)*1000000); //us
	cout << "running time for building Gp: " << run_time / 1000 << "ms\n";
	cout << "Edges in Gps: " << nonZeros(B) <<endl;

	gettimeofday(&time1,NULL);
	parallel_core_decomposition(B);
	gettimeofday(&time2,NULL);
	run_time= ((time2.tv_usec-time1.tv_usec)+(time2.tv_sec-time1.tv_sec)*1000000); //us
	cout << "running time for core decomposition: " << run_time / 1000 << "ms\n";

	gettimeofday(&time_over,NULL);
	run_time= ((time_over.tv_usec-time_start.tv_usec)+(time_over.tv_sec-time_start.tv_sec)*1000000); //us
	cout << "running time of BoolAPCore(D):" << run_time / 1000 << "ms\n";
	BoolAPCoreD_runtime = run_time;
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
	for(int i = 0; i < g.n; i++) core[i]=degree[i];
#pragma omp barrier
}

void initializeDegree(matrix Adjacent, int* degree, vector<int>& V, vector<neighbor>& neighbors)
{
	vector<int> empty_neighbors;
	for (int r = 0; r < g.n; r++)
	{
		neighbors.push_back(empty_neighbors);
		degree[r] = 0;
	}

#pragma omp parallel for num_threads(NUM_THREADS)
	for (int r = 0; r < Adjacent.nonzero_row.size(); r++)
	{
		int cur_row = Adjacent.nonzero_row[r];
		int bg = Adjacent.lhs_index[r];
		int ed = Adjacent.lhs_index[r + 1] - 1;
		degree[cur_row] = ed - bg;
		for (int element = bg; element <= ed; element++)
			if (Adjacent.lhs_elements[element] != cur_row)
			{
				neighbors[cur_row].push_back(Adjacent.lhs_elements[element]);
			}
#pragma omp critical
		{
			V.push_back(cur_row);
		}
	}
#pragma omp barrier
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
		Index empty_idx;
		vector <int> empty_elements;
		empty_idx = -1;
		for (int r = 0; r < g.n; r++) A.lhs_index.push_back(empty_idx);
		A.rhs_elements = empty_elements;
		for (int c = 0; c < g.n; c++) A.rhs_index.push_back(empty_idx);
		return;
	}

	sort(t.begin(), t.end(), cmp1); //lhs
	int row = t[0].i;
	Index idx;
	idx = 0;
	for (int element = 0; element < t.size(); element++)
	{
		A.lhs_elements.push_back(t[element].j);
		if (t[element].i > row)
		{
			A.nonzero_row.push_back(row);
			A.lhs_index.push_back(idx);
			idx = element;
			row = t[element].i;
		}
	}
	A.nonzero_row.push_back(row);
	A.lhs_index.push_back(idx); //index for last row
	idx = t.size(); //last index
	A.lhs_index.push_back(idx);


	sort(t.begin(), t.end(), cmp2); //rhs
	int col = t[0].j;
	idx = 0;
	for (int element = 0; element < t.size(); element++)
	{
		A.rhs_elements.push_back(t[element].i);
		if (t[element].j > col)
		{
			A.nonzero_col.push_back(col);
			A.rhs_index.push_back(idx);
			idx = element;
			col = t[element].j;
		}
	}
	A.nonzero_col.push_back(col);
	A.rhs_index.push_back(idx); //index for last column
	idx = t.size(); //last index
	A.rhs_index.push_back(idx);
}

matrix transpose(matrix A, int format) //format=0, lhs; format=1,rhs;
{
	matrix B;
	B.col = A.row;
	B.row = A.col;
	if (format == 0)
	{
		B.nonzero_row = A.nonzero_col;
		B.lhs_index = A.rhs_index;
		B.lhs_elements = A.rhs_elements;
	}
	else
	{
		B.nonzero_col = A.nonzero_row;
		B.rhs_index = A.lhs_index;
		B.rhs_elements = A.lhs_elements;
	}
	return B;
}

matrix boolProduct(matrix A, matrix B) //only need lhs format
{
	if (NUM_THREADS == 1) return non_parallel_boolProduct(A, B);

	matrix C;
	C.row = A.row;
	C.col = B.col;

	vector<int> empty_elements;
	vector<vector<int>> elements;

	for (int cur_row = 0; cur_row < C.row; cur_row++)
	{
		elements.push_back(empty_elements);
	}
	vector<vector<int>> thread_result(NUM_THREADS, vector<int>(C.col, 0));
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int r = 0; r < A.nonzero_row.size(); r++)
	{
		int tid = omp_get_thread_num();
		vector<int>& result = thread_result[tid];
		int cur_row = A.nonzero_row[r];
		int row_bg = A.lhs_index[r];
		int row_ed = A.lhs_index[r + 1] - 1;
		for (int element = row_bg; element <= row_ed; element++)
			result[A.lhs_elements[element]] = 1;
		for (int c = 0; c < B.nonzero_col.size(); c++) //compute (cur_row, cur_col)
		{
			int cur_col = B.nonzero_col[c];
			int col_bg = B.rhs_index[c];
			int col_ed = B.rhs_index[c + 1] - 1;
			for (int element = col_bg; element <= col_ed; element++)
			{
				if (result[B.rhs_elements[element]] == 1)
				{
					elements[cur_row].push_back(cur_col);
					break;
				}
			}
		}
		for (int element = row_bg; element <= row_ed; element++)
			result[A.lhs_elements[element]] = 0;
	}
#pragma omp barrier

	int last_size = 0, cur_size;
	Index idx;
	for (int cur_row = 0; cur_row < C.row; cur_row++)
	{
		int size = elements[cur_row].size();
		if (size > 0)
		{
			C.lhs_elements.insert(C.lhs_elements.end(), elements[cur_row].begin(), elements[cur_row].end());
			idx = last_size;
			cur_size = last_size + size;
			last_size = cur_size;
			C.nonzero_row.push_back(cur_row);
			C.lhs_index.push_back(idx);
		}
	}
	C.lhs_index.push_back(last_size);
	return C;
}

matrix boolTransposeProduct(matrix A)
{
	matrix C;
	C.col = C.row = A.col;

	vector<int> empty_elements;
	vector<vector<int>> elements;

	for (int cur_row = 0; cur_row < C.row; cur_row++)
	{
		elements.push_back(empty_elements);
	}
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int r = 0; r < A.nonzero_row.size(); r++)
	{
		int cur_row = A.nonzero_row[r];
		int row_bg = A.lhs_index[r];
		int row_ed = A.lhs_index[r + 1] - 1;
		bool* result = new bool[C.col];
		memset(result, 0, sizeof(bool) * C.col);
		for (int element = row_bg; element <= row_ed; element++)
			result[A.lhs_elements[element]] = 1;
		for (int c = 0; c < A.nonzero_row.size(); c++) //compute (cur_row, cur_col)
		{
			int cur_col = A.nonzero_row[c];
			if (cur_col > cur_row) break; //only compute half of the matrix
			int col_bg = A.lhs_index[c];
			int col_ed = A.lhs_index[c + 1] - 1;
			for (int element = col_bg; element <= col_ed; element++)
			{
				if (result[A.lhs_elements[element]] == 1)
				{
					elements[cur_row].push_back(cur_col);
					break;
				}
			}
		}
		delete[]result;
	}
#pragma omp barrier

	for (int cur_row = 0; cur_row < C.row; cur_row++)
	{
		int size = elements[cur_row].size();
		if (size > 0)
		{
			for (int id = 0; id < size - 1; id++)
			{
				elements[elements[cur_row][id]].push_back(cur_row);
			}
		}
	}

	int last_size = 0, cur_size;
	Index idx;
	for (int cur_row = 0; cur_row < C.row; cur_row++)
	{
		int size = elements[cur_row].size();
		if (size > 0)
		{
			C.lhs_elements.insert(C.lhs_elements.end(), elements[cur_row].begin(), elements[cur_row].end());
			idx = last_size;
			cur_size = last_size + size;
			last_size = cur_size;
			C.nonzero_row.push_back(cur_row);
			C.lhs_index.push_back(idx);
		}
	}
	C.lhs_index.push_back(last_size);
	return C;
}

matrix non_parallel_boolProduct(matrix A, matrix B) //only need lhs format
{
	matrix C;
	C.row = A.row;
	C.col = B.col;
	bool* result = new bool[C.col];
	memset(result, 0, sizeof(bool) * C.col);
	int last_size = 0, cur_size;
	Index idx;
	for (int r = 0; r < A.nonzero_row.size(); r++)
	{
		int cur_row = A.nonzero_row[r];
		int row_bg = A.lhs_index[r];
		int row_ed = A.lhs_index[r + 1] - 1;
		vector<int> nnz = {};
		for (int element = row_bg; element <= row_ed; element++)
		{
			result[A.lhs_elements[element]] = 1;
			nnz.push_back(A.lhs_elements[element]);
		}
		for (int c = 0; c < B.nonzero_col.size(); c++) //compute (cur_row, cur_col)
		{
			int cur_col = B.nonzero_col[c];
			int col_bg = B.rhs_index[c];
			int col_ed = B.rhs_index[c + 1] - 1;
			for (int element = col_bg; element <= col_ed; element++)
			{
				if (result[B.rhs_elements[element]] == 1)
				{
					C.lhs_elements.push_back(cur_col);
					break;
				}
			}
		}
		cur_size = C.lhs_elements.size();
		if (cur_size != last_size)
		{
			idx = last_size;
			C.nonzero_row.push_back(cur_row);
			C.lhs_index.push_back(idx);
			last_size = cur_size;
		}
		for (int element = 0; element < nnz.size(); element++)
		{
			result[nnz[element]] = 0;
		}
	}
	C.lhs_index.push_back(last_size);
	return C;
}

int nonZeros(matrix A)
{
	return A.lhs_elements.size();
}