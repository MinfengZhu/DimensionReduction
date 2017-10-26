#ifndef KNN_H
#define KNN_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

using namespace std;

typedef float real;

class data{
private:
	void clean_data();
public:
	real *vec;
	long long n_vertices, n_dim;
	data();
	real* load_from_file(char *infile);
    long long get_vertice_number();
    long long get_dim_number();
    void show_data_info();
};

#endif
