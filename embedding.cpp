#ifndef EMBEDDING
#define EMBEDDING

#include "embedding.h"
#include <map>
#include <float.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <fstream>
#include <malloc.h>
#include <string>

embedding::embedding() {}

const gsl_rng_type *embedding::gsl_T = NULL;
gsl_rng *embedding::gsl_r = NULL;

void embedding::compute_similarity()
{
    printf("[EMBEDDING] Computing similarities ......"); fflush(stdout);
    n_edge = 0;
    head = new long long[n_vertices];
    long long i, x, y, p, q;
    real sum_weight = 0;
    for (i = 0; i < n_vertices; ++i) head[i] = -1;
    for (x = 0; x < n_vertices; ++x)
    {
        for (i = 0; i < knn_vec[x].size(); ++i)
        {
            edge_from.push_back((int)x);
            edge_to.push_back((int)(y = knn_vec[x][i]));
            edge_weight.push_back(CalcDist(x, y));
            next.push_back(head[x]);
            reverse.push_back(-1);
            head[x] = n_edge++;
        }
    }
    delete[] vec; vec = NULL;
    delete[] knn_vec; knn_vec = NULL;
    pthread_t *pt = new pthread_t[n_threads];
    for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, embedding::compute_similarity_thread_caller, new arg_embedding(this, j));
    for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
    delete[] pt;

    pt = new pthread_t[n_threads];
    for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, embedding::search_reverse_thread_caller, new arg_embedding(this, j));
    for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
    delete[] pt;

    for (x = 0; x < n_vertices; ++x)
    {
        for (p = head[x]; p >= 0; p = next[p])
        {
            y = edge_to[p];
            q = reverse[p];
            if (q == -1)
            {
                edge_from.push_back((int)y);
                edge_to.push_back((int)x);
                edge_weight.push_back(0);
                next.push_back(head[y]);
                reverse.push_back(p);
                q = reverse[p] = head[y] = n_edge++;
            }
            if (x > y){
                sum_weight += edge_weight[p] + edge_weight[q];
                edge_weight[p] = edge_weight[q] = (edge_weight[p] + edge_weight[q]) / 2;
            }
        }
    }
    printf(" Done.\n");
}

real embedding::CalcDist(long long x, long long y)
{
    real ret = 0;
    long long i, lx = x * n_dim, ly = y * n_dim;
    for (i = 0; i < n_dim; ++i)
        ret += (vec[lx + i] - vec[ly + i]) * (vec[lx + i] - vec[ly + i]);
    return ret;
}

void *embedding::search_reverse_thread_caller(void *arg)
{
    embedding *ptr = (embedding*)(((arg_embedding*)arg)->ptr);
    ptr->search_reverse_thread(((arg_embedding*)arg)->id);
    pthread_exit(NULL);
}

void embedding::compute_similarity_thread(int id)
{
    long long lo = id * n_vertices / n_threads;
    long long hi = (id + 1) * n_vertices / n_threads;
    long long x, iter, p;
    real beta, lo_beta, hi_beta, sum_weight, H, tmp;
    for (x = lo; x < hi; ++x)
    {
        beta = 1;
        lo_beta = hi_beta = -1;
        for (iter = 0; iter < 200; ++iter)
        {
            H = 0;
            sum_weight = FLT_MIN;
            for (p = head[x]; p >= 0; p = next[p])
            {
                sum_weight += tmp = exp(-beta * edge_weight[p]);
                H += beta * (edge_weight[p] * tmp);
            }
            H = (H / sum_weight) + log(sum_weight);
            if (fabs(H - log(perplexity)) < 1e-5) break;
            if (H > log(perplexity))
            {
                lo_beta = beta;
                if (hi_beta < 0) beta *= 2; else beta = (beta + hi_beta) / 2;
            }
            else{
                hi_beta = beta;
                if (lo_beta < 0) beta /= 2; else beta = (lo_beta + beta) / 2;
            }
            if(beta > FLT_MAX) beta = FLT_MAX;
        }
        for (p = head[x], sum_weight = FLT_MIN; p >= 0; p = next[p])
        {
            sum_weight += edge_weight[p] = exp(-beta * edge_weight[p]);
        }
        for (p = head[x]; p >= 0; p = next[p])
        {
            edge_weight[p] /= sum_weight;
        }
    }
}

void *embedding::compute_similarity_thread_caller(void *arg)
{
    embedding *ptr = (embedding*)(((arg_embedding*)arg)->ptr);
    ptr->compute_similarity_thread(((arg_embedding*)arg)->id);
    pthread_exit(NULL);
}

void embedding::search_reverse_thread(int id)
{
    long long lo = id * n_vertices / n_threads;
    long long hi = (id + 1) * n_vertices / n_threads;
    long long x, y, p, q;
    for (x = lo; x < hi; ++x)
    {
        for (p = head[x]; p >= 0; p = next[p])
        {
            y = edge_to[p];
            for (q = head[y]; q >= 0; q = next[q])
            {
                if (edge_to[q] == x) break;
            }
            reverse[p] = q;
        }
    }
}

void embedding::init_neg_table()
{
	long long x, p, i;
	neg_size = 1e8;
	reverse.clear(); vector<long long> (reverse).swap(reverse);
	real sum_weights = 0, dd, *weights = new real[n_vertices];
	for (i = 0; i < n_vertices; ++i) weights[i] = 0;
	for (x = 0; x < n_vertices; ++x)
	{
		for (p = head[x]; p >= 0; p = next[p])
		{
			weights[x] += edge_weight[p];
		}
		sum_weights += weights[x] = pow(weights[x], 0.75);
	}
	next.clear(); vector<long long> (next).swap(next);
	delete[] head; head = NULL;
	neg_table = new int[neg_size];
	dd = weights[0];
	for (i = x = 0; i < neg_size; ++i)
	{
		neg_table[i] = x;
		if (i / (real)neg_size > dd / sum_weights && x < n_vertices - 1)
		{
			dd += weights[++x];
		}
	}
	delete[] weights;
}

void embedding::init_alias_table()
{
    alias = new long long[n_edge];
    prob = new real[n_edge];

    real *norm_prob = new real[n_edge];
    long long *large_block = new long long[n_edge];
    long long *small_block = new long long[n_edge];

    real sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for (long long k = 0; k < n_edge; ++k) sum += edge_weight[k];
    for (long long k = 0; k < n_edge; ++k) norm_prob[k] = edge_weight[k] * n_edge / sum;

    for (long long k = n_edge - 1; k >= 0; --k)
    {
        if (norm_prob[k] < 1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob[cur_small_block] = norm_prob[cur_small_block];
        alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;

    delete[] norm_prob;
    delete[] small_block;
    delete[] large_block;
}

long long embedding::sample_an_edge(real rand_value1, real rand_value2)
{
    long long k = (long long)((n_edge - 0.1) * rand_value1);
    return rand_value2 <= prob[k] ? k : alias[k];
}


void embedding::visualize_thread(int id)
{
	long long edge_count = 0, last_edge_count = 0;
	long long x, y, p, lx, ly, i, j;
	real f, g, gg, cur_alpha = initial_alpha;
	real *cur = new real[out_dim];
	real *err = new real[out_dim];
	real grad_clip = 5.0;
	while (1)
	{
		if (edge_count > n_samples / n_threads + 2) break;
		if (edge_count - last_edge_count > 10000)
		{
			edge_count_actual += edge_count - last_edge_count;
			last_edge_count = edge_count;
			cur_alpha = initial_alpha * (1 - edge_count_actual / (n_samples + 1.0));
			if (cur_alpha < initial_alpha * 0.0001) cur_alpha = initial_alpha * 0.0001;
			printf("%c[EMBEDDING] Fitting model\tAlpha: %f Progress: %.3lf%%", 13, cur_alpha, (real)edge_count_actual / (real)(n_samples + 1) * 100);
			fflush(stdout);
		}
		p = sample_an_edge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
		x = edge_from[p];
		y = edge_to[p];
		lx = x * out_dim;
		for (i = 0; i < out_dim; ++i) cur[i] = vis[lx + i], err[i] = 0;
		for (i = 0; i < n_negatives + 1; ++i)
		{
			if (i > 0)
			{
				y = neg_table[(unsigned long long)floor(gsl_rng_uniform(gsl_r) * (neg_size - 0.1))];
				if (y == edge_to[p]) continue;
			}
			ly = y * out_dim;
			for (j = 0, f= 0; j < out_dim; ++j) f += (cur[j] - vis[ly + j]) * (cur[j] - vis[ly + j]);
			if (i == 0) g = -2 / (1 + f);
			else g = 2 * gamma / (1 + f) / (0.1 + f);
			for (j = 0; j < out_dim; ++j)
			{
				gg = g * (cur[j] - vis[ly + j]);
				if (gg > grad_clip) gg = grad_clip;
				if (gg < -grad_clip) gg = -grad_clip;
				err[j] += gg * cur_alpha;

				gg = g * (vis[ly + j] - cur[j]);
				if (gg > grad_clip) gg = grad_clip;
				if (gg < -grad_clip) gg = -grad_clip;
				vis[ly + j] += gg * cur_alpha;
			}
		}
		for (int j = 0; j < out_dim; ++j) vis[lx + j] += err[j];
		++edge_count;
	}
	delete[] cur;
	delete[] err;
}

void *embedding::visualize_thread_caller(void *arg)
{
	embedding *ptr = (embedding*)(((arg_embedding*)arg)->ptr);
	ptr->visualize_thread(((arg_embedding*)arg)->id);
	pthread_exit(NULL);
}

void embedding::load_knn(data &d, real perp, long long n_thre, long long out_d, real alph, long long n_samp, long long n_nega, real gamm) {
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);

    vec = d.getVec();
    knn_vec = d.getKnnVec();
    n_vertices = d.get_vertice_number();
    n_dim = d.get_dim_number();

    if (!vec && !knn_vec)
    {
        printf("[EMBEDDING] Missing knn data!\n");
        return;
    }
    out_dim = out_d < 0 ? 2 : out_d;
    initial_alpha = alph < 0 ? 1.0 : alph;
    n_threads = n_thre < 0 ? 8 : n_thre;
    n_samples = n_samp;
    n_negatives = n_nega < 0 ? 5 : n_nega;
    gamma = gamm < 0 ? 7.0 : gamm;
    perplexity = perp < 0 ? 50.0 : perp;

    if (n_samples < 0)
    {
        if (n_vertices < 10000)
            n_samples = 1000;
        else if (n_vertices < 1000000)
            n_samples = (n_vertices - 10000) * 9000 / (1000000 - 10000) + 1000;
        else n_samples = n_vertices / 100;
    }
    n_samples *= 1000000;
}

void embedding::visualize()
{
    compute_similarity();
	long long i;
	//init projection coordinate
	vis = new real[n_vertices * out_dim];
	for (i = 0; i < n_vertices * out_dim; ++i) vis[i] = (gsl_rng_uniform(gsl_r) - 0.5) / out_dim * 0.0001;
	init_neg_table();
	init_alias_table();
	edge_count_actual = 0;
	pthread_t *pt = new pthread_t[n_threads];
	for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, embedding::visualize_thread_caller, new arg_embedding(this, j));
	for (int j = 0; j < n_threads; ++j) pthread_join(pt[j], NULL);
	delete[] pt;
	printf("\n");
}

void embedding::save(char *outfile)
{
    FILE *fout = fopen(outfile, "wb");
    fprintf(fout, "%lld %lld\n", n_vertices, out_dim);
    for (long long i = 0; i < n_vertices; ++i)
    {
        for (long long j = 0; j < out_dim; ++j)
        {
            if (j) fprintf(fout, " ");
            fprintf(fout, "%.6f", vis[i * out_dim + j]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
}

#endif
