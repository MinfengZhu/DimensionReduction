#ifndef KNN_H
#define KNN_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "./data.h"

#include "ANNOY/annoylib.h"
#include "ANNOY/kissrandom.h"

#include <pthread.h>
#include <gsl/gsl_rng.h>

#include "algorithm/efanna.hpp"

using namespace std;
using namespace efanna;

typedef float real;

struct arg_struct{
    void *ptr;
    int id;
    arg_struct(void *x, int y) :ptr(x), id(y){}
};

class knn{
private:
    long long n_dim, n_vertices, n_threads, n_trees, n_neighbors, n_propagations;
    real *vec;
    vector<int> *knn_vec, *old_knn_vec;
    AnnoyIndex<int, real, Euclidean, Kiss64Random> *annoy_index;
    static const gsl_rng_type * gsl_T;
    static gsl_rng * gsl_r;

    string knn_type;

    // efanna建立knn需要的参数
    int knn_trees, epochs, mlevel, L, checkK, knn_k, S, build_trees;

    void run_annoy();
    void annoy_thread(int id);
    static void *annoy_thread_caller(void *arg);
    void run_propagation();
    void propagation_thread(int id);
    static void *propagation_thread_caller(void *arg);
    void test_accuracy();
    real CalcDist(long long x, long long y);

public:
	knn();
	void setParams(data& d, long long n_tree, long long n_neig, long long n_thre, long long n_prop,
                   int knn_tre, int epo, int mle, int l, int che, int k, int s, int build_tre, string knn_tp);
    vector<int>* construct_knn();
};

#endif
