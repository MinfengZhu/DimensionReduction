#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "data.h"

#include <pthread.h>
#include <gsl/gsl_rng.h>

using namespace std;

typedef float real;

struct arg_embedding{
    void *ptr;
    int id;
    arg_embedding(void *x, int y) :ptr(x), id(y){}
};

class embedding{
private:
    long long n_edge, *head, n_vertices, n_dim, n_threads, out_dim, edge_count_actual, n_samples, n_negatives,
            neg_size, *alias;
    vector<int> edge_from, edge_to, *knn_vec;
    vector<real> edge_weight;
    vector<long long> next, reverse;
    static const gsl_rng_type * gsl_T;
    static gsl_rng * gsl_r;
    int *neg_table;

    real perplexity, *vec, *vis, initial_alpha, gamma, *prob;

    void compute_similarity();
    void compute_similarity_thread(int id);
    static void *compute_similarity_thread_caller(void *arg);
    void search_reverse_thread(int id);
    static void *search_reverse_thread_caller(void *arg);
    void init_alias_table();
	void init_neg_table();
	void visualize_thread(int id);
	static void *visualize_thread_caller(void *arg);
    real CalcDist(long long x, long long y);
    long long sample_an_edge(real rand_value1, real rand_value2);

public:
    embedding();
    void visualize();
    void load_knn(data &d, real perp, long long n_thre, long long out_d, real alph, long long n_samp, long long n_nega, real gamm);
    void save(char *outfile);
};

#endif
