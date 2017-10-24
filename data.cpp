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

//Transfer data
data::data()
{
	vec = NULL;
}

void data::clean_data()
{
	if (vec) { delete[] vec; vec = NULL; }
}

real* data::load_from_file(char *infile, long long &n_vertices, long long &n_dim)
{
	clean_data();
	FILE *fin = fopen(infile, "rb");
	if (fin == NULL)
	{
		printf("\nFile not found!\n");
		return 0;
	}
    printf("Reading input file %s ......", infile); fflush(stdout);
	auto ret = fscanf(fin, "%lld%lld", &n_vertices, &n_dim);
	//n_vertices=10000;
	if (ret == EOF){
		printf("Reading error");
		return 0;
	}
	vec = new float[n_vertices * n_dim];
	for (long long i = 0; i < n_vertices; ++i)
	{
		for (long long j = 0; j < n_dim; ++j)
		{
			auto ret = fscanf(fin, "%f", &vec[i * n_dim + j]);
			if (ret == EOF){
				printf("Reading error"); 
				return 0;
			}
		}
	}
	fclose(fin);
	printf(" Done.\n");
	printf("Total vertices : %lld\tDimension : %lld\n", n_vertices, n_dim);
	return vec;
}

#endif