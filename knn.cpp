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
#include <stdio.h>

knn::knn () {
}

const gsl_rng_type *knn::gsl_T = NULL;
gsl_rng *knn::gsl_r = NULL;

void knn::setParams (data& d, long long n_tree, long long n_neig, long long n_thre, long long n_prop,
                     int knn_tre, int epo, int mle, int l, int che, int k, int s, int build_tre, string knn_tp) {
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);

    vec = d.getVec();
    knn_vec = d.getKnnVec();
    n_vertices = d.get_vertice_number();
    n_dim = d.get_dim_number();
    knn_type = knn_tp;
    // largevis knn 需要的参数
    n_threads = n_thre < 0 ? 8 : n_thre;
    n_trees = n_tree;
    n_neighbors = n_neig < 0 ? 150 : n_neig;
    n_propagations = n_prop < 0 ? 3 : n_prop;
//    perplexity = perp < 0 ? 50.0 : perp;

    if (n_trees < 0)
    {
        if (n_vertices < 100000)
            n_trees = 10;
        else if (n_vertices < 1000000)
            n_trees = 20;
        else if (n_vertices < 5000000)
            n_trees = 50;
        else n_trees = 100;
    }

    // efanna knn 需要的参数
    knn_trees = knn_tre < 0 ? 8 : knn_tre;
    mlevel = mle < 0 ? 8 : mle;
    epochs = epo < 0 ? 8 : epo;
    L = l < 0 ? 30 : l;
    checkK = che < 0 ? 25 : che;
    knn_k = k < 0 ? 10 : k;
    S = s < 0 ? 10 : s;
    build_trees = build_tre < 0 ? 8 : build_tre;
}

vector<int>* knn::construct_knn () {
    if (knn_type.empty()) {
        cout << "[kNN Graph] " << "no specified the type of constructing knn." << endl;
    } else {
        if (knn_type == "largevis") {
            run_annoy();
            run_propagation();
        }else if (knn_type == "efanna") {
            Matrix<float> dataset(n_vertices, n_dim, vec);
            //generate knn graph
            FIndex<float> index(dataset, new L2DistanceAVX<float>(), efanna::KDTreeUbIndexParams(true, knn_trees, mlevel, epochs, checkK, L, knn_k, build_trees, S));
            clock_t s, f;
            time_t time_s,time_e;
            s = clock();//start time
            time(&time_s);
            index.buildIndex();
            f = clock();//end time
            time(&time_e);
            cout<<"[kNN Graph] " << "Efanna kNN building CPU time : " << (f - s) * 1.0 / CLOCKS_PER_SEC << " seconds" << endl;
            cout<< "[kNN Graph] " << "EFanna kNN building real time: " << difftime(time_e, time_s) << " seconds"<<endl;

            index.getGraphResult(knn_vec);
            //for (int step = 0; step < 10; step++) {
            //    test_accuracy2(step * 50000);
            //}
            //index.testGsAccuracy(dataset, knn_k);

        }
        test_accuracy();
        return knn_vec;
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
    printf("[kNN Graph] Running ANNOY ......"); fflush(stdout);
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
        printf("[kNN Graph] Running propagation %d/%d%c", (int)i + 1, (int)n_propagations, 13);
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

void knn::test_accuracy()
{
    long long test_case = 1000;
    std::priority_queue< pair<real, int> > *heap = new std::priority_queue< pair<real, int> >;
    long long hit_case = 0, i, j, x, y;
    for (i = 0; i < test_case; ++i)
    {
        x = floor(gsl_rng_uniform(gsl_r) * (n_vertices - 0.1));
        for (y = 0; y < n_vertices; ++y) if (x != y)
            {
                heap->push(std::make_pair(CalcDist(x, y), y));
                if (heap->size() == n_neighbors + 1) heap->pop();
            }
        while (!heap->empty())
        {
            y = heap->top().second;
            heap->pop();
            for (j = 0; j < knn_vec[x].size(); ++j) if (knn_vec[x][j] == y)
                    ++hit_case;
        }
    }
    delete heap;
    heap = NULL;
    printf("[kNN Graph] Test efanna knn accuracy(use largevis test) : %.2f%%\n", hit_case * 100.0 / (test_case * n_neighbors));
}

void knn::save_knn(char* outfile){
    ofstream myfile;
    myfile.open (outfile);
    //FILE *fout = fopen(outfile, "wb");
    //fprintf(fout, "%lld %lld\n", n_vertices, out_dim);
    if(myfile.is_open()){


    for (int i = 0; i < n_vertices; ++i){
        for(int j=0; j < knn_vec[i].size(); j++){
            //if (j) fprintf(fout, " ");
            //std::print(fout,"%d", knn_vec[i][j]);
            if(j) myfile<< " ";
            myfile<< knn_vec[i][j];
        }
        myfile<<"\n";
        //fprintf(fout, "\n");
    }
    myfile<< flush;
    myfile.close();
    //fclose(fout);

    }
    else
        cout<< "unable to write file"<<endl;
}
#endif
