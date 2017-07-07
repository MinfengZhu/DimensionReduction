//
// Created by twilightsnow on 17-7-6.
//

#ifndef VIS_HELLOCUDA_H
#define VIS_HELLOCUDA_H
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <time.h>
void show(double *initial_alpha,long long *out_dim,long long *n_edge,double *prob,long long *alias,
          double *vis_d,long long *n_vertices, long long *n_samples, int *edge_from, int *edge_to, int edge_ft_size, long long n_negatives, int *neg_table, long long neg_size);
#endif //VIS_HELLOCUDA_H
