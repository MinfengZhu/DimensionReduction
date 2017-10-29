#ifndef DATA
#define DATA

#include "data.h"
#include <map>
#include <float.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <fstream>
#include <malloc.h>
#include <string>

data::data()
{
	vec = NULL;
}

void data::clean_data()
{
	if (vec) { delete[] vec; vec = NULL; }
}

real* data::load_from_file(char *infile)
{
	clean_data();
	FILE *fin = fopen(infile, "rb");
	if (fin == NULL)
	{
		printf("\nFile not found!\n");
		return 0;
	}
    cout<<"[DATA] "<<"Read from "<<infile<<endl;
	auto ret = fscanf(fin, "%lld%lld", &n_vertices, &n_dim);
    if (ret == EOF){
		printf("Reading error");
		return 0;
	}
	vec = new float[n_vertices * n_dim];
    knn_vec = new vector<int>[n_vertices];
	for (long long i = 0; i < n_vertices; ++i){
		for (long long j = 0; j < n_dim; ++j){
            auto ret = fscanf(fin, "%f", &vec[i * n_dim + j]);
			if (ret == EOF){
				printf("Reading error");
				return 0;
			}
		}
	}
//    cout << "[TEST PRE ] " << vec[176] << endl;
	fclose(fin);
	show_data_info();
	normalize();
//    cout << "[TEST AFTER ] " << vec[176] << endl;
    return vec;
}

long long data::get_vertice_number(){
    return n_vertices;
}

long long data::get_dim_number(){
    return n_dim;
}
real* data::getVec(){
	return vec;
}
void data::show_data_info(){
    cout<<"[DATA] "<<"#vertices="<<n_vertices<<" "<<"#dim="<<n_dim<<endl;
}

void data::normalize() {
	cout << "[DATA] " << "Normalizing ......";
	real *mean = new real[n_dim];
	for (long long i = 0; i < n_dim; ++i) mean[i] = 0;
	for (long long i = 0, ll = 0; i < n_vertices; ++i, ll += n_dim)
	{
		for (long long j = 0; j < n_dim; ++j)
			mean[j] += vec[ll + j];
	}
	for (long long j = 0; j < n_dim; ++j)
		mean[j] /= n_vertices;
	real mX = 0;
	for (long long i = 0, ll = 0; i < n_vertices; ++i, ll += n_dim)
	{
		for (long long j = 0; j < n_dim; ++j)
		{
			vec[ll + j] -= mean[j];
			if (fabs(vec[ll + j]) > mX)	mX = fabs(vec[ll + j]);
		}
	}
	for (long long i = 0; i < n_vertices * n_dim; ++i)
		vec[i] /= mX;
	delete[] mean;
	cout << " Done." << endl;
}

vector<int>* data::getKnnVec () {
	return knn_vec;
}

#endif
