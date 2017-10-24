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
	void clean_data();
public:
	real *vec;
	long long n_vertices, n_dim;
	data();
	real* load_from_file(char *infile,long long &n_vertices, long long &n_dim);
};

#endif