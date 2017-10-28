#ifndef KNN
#define KNN

#include "knn.h"
#include <map>
#include <float.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <fstream>
#include <malloc.h>
#include <string>

knn::knn () {
}

void knn::setParams (data& d, long long n_tree, long long n_nega, long long n_thre) {
    vec = d.getVec();
    knn_vec = d.getKnnVec();
    n_vertices = d.get_vertice_number();
    n_dim = d.get_dim_number();
    n_threads = n_thre;
    n_trees = n_tree;
    n_neighbors = n_neig;
}

void knn::construct_knn (string knn_type) {
    if (knn_type.empty()) {
        cout << "[KNN] " << "no specified the type of constructing knn." << endl;
    } else {
        if (knn_type == "largevis") {
            run_annoy();
            run_propagation();
        }else if (knn_type == "efanna") {

        }
        test_accuracy();
        compute_similarity();
    }
}

void *knn::annoy_thread_caller(void *arg)
{
    knn *ptr = (knn*)(((arg_struct*)arg)->ptr);
    ptr->annoy_thread(((arg_struct*)arg)->id);
    pthread_exit(NULL);
}

void knn::annoy_thread(int id)
{
    long long lo = id * n_vertices / n_threads;
    long long hi = (id + 1) * n_vertices / n_threads;
    AnnoyIndex<int, real, Euclidean, Kiss64Random> *cur_annoy_index = NULL;
    if (id > 0)
    {
        cur_annoy_index = new AnnoyIndex<int, real, Euclidean, Kiss64Random>(n_dim);
        cur_annoy_index->load("annoy_index_file");
    }
    else
        cur_annoy_index = annoy_index;
    for (long long i = lo; i < hi; ++i)
    {
        cur_annoy_index->get_nns_by_item(i, n_neighbors + 1, (n_neighbors + 1) * n_trees, &knn_vec[i], NULL);
        for (long long j = 0; j < knn_vec[i].size(); ++j)
            if (knn_vec[i][j] == i)
            {
                knn_vec[i].erase(knn_vec[i].begin() + j);
                break;
            }
    }
    if (id > 0) delete cur_annoy_index;
}

void knn::run_annoy() {
    printf("Running ANNOY ......"); fflush(stdout);
    annoy_index = new AnnoyIndex<int, real, Euclidean, Kiss64Random>(n_dim);
    for (long long i = 0; i < n_vertices; ++i)
        annoy_index->add_item(i, &vec[i * n_dim]);
    annoy_index->build(n_trees);
    if (n_threads > 1) annoy_index->save("annoy_index_file");
    knn_vec = new std::vector<int>[n_vertices];

    pthread_t *pt = new pthread_t[n_threads];
    for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, knn::annoy_thread_caller, new arg_struct(this, j));
    for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
    delete[] pt;
    delete annoy_index; annoy_index = NULL;
    printf(" Done.\n");
}

void knn::propagation_thread(int id)
{
    long long lo = id * n_vertices / n_threads;
    long long hi = (id + 1) * n_vertices / n_threads;
    int *check = new int[n_vertices];
    std::priority_queue< pair<real, int> > heap;
    long long x, y, i, j, l1, l2;
    for (x = 0; x < n_vertices; ++x) check[x] = -1;
    for (x = lo; x < hi; ++x)
    {
        check[x] = x;
        std::vector<int> &v1 = old_knn_vec[x];
        l1 = v1.size();
        for (i = 0; i < l1; ++i)
        {
            y = v1[i];
            check[y] = x;
            heap.push(std::make_pair(CalcDist(x, y), y));
            if (heap.size() == n_neighbors + 1) heap.pop();
        }
        for (i = 0; i < l1; ++i)
        {
            std::vector<int> &v2 = old_knn_vec[v1[i]];
            l2 = v2.size();
            for (j = 0; j < l2; ++j) if (check[y = v2[j]] != x)
                {
                    check[y] = x;
                    heap.push(std::make_pair(CalcDist(x, y), (int)y));
                    if (heap.size() == n_neighbors + 1) heap.pop();
                }
        }
        while (!heap.empty())
        {
            knn_vec[x].push_back(heap.top().second);
            heap.pop();
        }
    }
    delete[] check;
}

void *knn::propagation_thread_caller(void *arg)
{
    knn *ptr = (knn*)(((arg_struct*)arg)->ptr);
    ptr->propagation_thread(((arg_struct*)arg)->id);
    pthread_exit(NULL);
}

void knn::run_propagation()
{
    for (int i = 0; i < n_propagations; ++i)
    {
        printf("Running propagation %d/%d%c", (int)i + 1, (int)n_propagations, 13);
        fflush(stdout);
        old_knn_vec = knn_vec;
        knn_vec = new std::vector<int>[n_vertices];
        pthread_t *pt = new pthread_t[n_threads];
        for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, knn::propagation_thread_caller, new arg_struct(this, j));
        for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
        delete[] pt;
        delete[] old_knn_vec;
        old_knn_vec = NULL;
    }
    printf("\n");
}


real knn::CalcDist(long long x, long long y)
{
    real ret = 0;
    long long i, lx = x * n_dim, ly = y * n_dim;
    for (i = 0; i < n_dim; ++i)
        ret += (vec[lx + i] - vec[ly + i]) * (vec[lx + i] - vec[ly + i]);
    return ret;
}


#endif
