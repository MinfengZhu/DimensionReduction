#ifndef DATA_H
#define DATA_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

using namespace std;

typedef float real;

class data{
private:
	real *vec;
	vector<int> *knn_vec;
	long long n_vertices, n_dim;
	void normalize();
	void clean_data();
public:
	data();
	real* load_from_file(char *infile);
    long long get_vertice_number();
	long long get_dim_number();
	real* getVec();
	void show_data_info();
	vector<int>* getKnnVec();
};

#endif
