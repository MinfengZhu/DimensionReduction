#ifndef KNN_H
#define KNN_H

#include <cstdlib>
#include <cstdio>
#include <string>
#include <cmath>
#include <vector>
#include "./data.h"

#include "ANNOY/annoylib.h"
#include "ANNOY/kissrandom.h"

using namespace std;

typedef float real;

struct arg_struct{
    void *ptr;
    int id;
    arg_struct(void *x, int y) :ptr(x), id(y){}
};

class knn{
private:
    long long n_dim, n_vertices, n_threads, n_trees, n_neighbors, n_propagations;
    float *vec;
    vector<int> *knn_vec, *old_knn_vec;
    AnnoyIndex<int, real, Euclidean, Kiss64Random> *annoy_index;

    void run_annoy();
    void annoy_thread(int id);
    static void *annoy_thread_caller(void *arg);
    void run_propagation();
    void propagation_thread(int id);
    static void *propagation_thread_caller(void *arg);
    void knn_largevis();
    void knn_efanna();
    void test_accuracy();
    void compute_similarity();
    real CalcDist(long long x, long long y);

public:
	knn();
	void setParams(data& d);
    vector<int>* construct_knn(string knn_type);
};

#endif
