#include <stdio.h>
#include "hellocuda.h"

const int blocksize = 1;

__global__ void hello(double *initial_alpha, long long *out_dim, long long *n_edge, double *vis, double* prob, long long* alias, long* edge_count_actual, long long * n_samples, int *edge_from, int *edge_to, long long *n_negatives,int *neg_table,long long *neg_size) {
    int id = threadIdx.x+ blockIdx.x * 1000;
    long edge_count = 0, last_edge_count = 0;
    long long x=1, y, p, lx, ly, i, j, k;
    double f, g, gg, cur_alpha = initial_alpha[0];
    double *cur = new double[out_dim[0]];
    double *err = new double[out_dim[0]];
    double gamma = 7;
    double grad_clip = 5.0;
    curandState state;
    curand_init(clock64(), id, 0, &state);
    while (1) {
        if (edge_count > *n_samples / 300/1000/2 + 2) {
            //for (int j = 0; j < *out_dim; ++j) vis[id] = *n_samples / 100/1000 + 2;
            break;
        }
//        if (edge_count > 1000) {
//            //for (int j = 0; j < *out_dim; ++j) vis[id] = ly+j;
//            //for (int j = 0; j < *out_dim; ++j) vis[id] = id;
//            break;
//        }
//        if (edge_count - last_edge_count > 10)
//        {
            //edge_count_actual[0] += edge_count - last_edge_count;
            //last_edge_count = edge_count;
            //cur_alpha = *initial_alpha * (1 - *edge_count_actual / (*n_samples + 1.0));
            cur_alpha = *initial_alpha * (1 - edge_count / (*n_samples *1.0 / 300/1000/2 + 10.0));
            if (cur_alpha < *initial_alpha * 0.0001) cur_alpha = *initial_alpha * 0.0001;
//        }


        k = (long long) ((n_edge[0] - 0.1) * curand_uniform(&state));
        p = curand_uniform(&state) <= prob[k] ? k : alias[k];
        x = edge_from[p];
        y = edge_to[p];
        lx = x * (out_dim[0]);
        for (i = 0; i < (*out_dim); ++i) cur[i] = vis[lx + i], err[i] = 0;
            for (i = 0; i < *n_negatives + 1; ++i)
            {
                if (i > 0)
                {

                    y = neg_table[(unsigned long long)floor(curand_uniform(&state) * (*neg_size - 0.1))];
                    if (y == edge_to[p]) continue;
                }
                ly = y * (*out_dim);
                for (j = 0, f= 0; j < (*out_dim); ++j) f += (cur[j] - vis[ly + j]) * (cur[j] - vis[ly + j]);
                if (i == 0) g = -2 / (1 + f);
                else g = 2 * gamma / (1 + f) / (0.1 + f);
                for (j = 0; j < out_dim[0]; ++j)
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
        for (int j = 0; j < *out_dim; ++j) vis[lx + j] += err[j];
        ++edge_count;
    }
//    vis[id] = 1 - edge_count / (*n_samples *1.0 / 100/1000/2 + 10.0);
//    vis[0] = edge_count;
//    vis[1] = *n_samples *1.0/100/1000/2 + 10.0;
    delete[] cur;
    delete[] err;
}

bool InitCUDA() {
    int count;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device./n");
        return false;
    }

    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x./n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}

void show(double *initial_alpha,long long *out_dim,long long *n_edge,double *prob,long long *alias,
          double *vis_d, long long *n_vertices, long long *n_samples, int *edge_from, int *edge_to, int edge_ft_size,long long n_negatives,int *neg_table, long long neg_size)
{
    if(InitCUDA())
        printf("CUDA initialized.\r\n");

    long long  *out_dim_device;
    double *initial_alpha_device;
    cudaMalloc( (void**)&initial_alpha_device, sizeof(double));
    cudaMalloc( (void**)&out_dim_device, sizeof(long long ) );
    cudaMemcpy( initial_alpha_device, initial_alpha, sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( out_dim_device, out_dim, sizeof(long long ), cudaMemcpyHostToDevice );

    long long *n_edge_device;
    cudaMalloc( (void**)&n_edge_device, sizeof(long long ));
    cudaMemcpy( n_edge_device, n_edge, sizeof(long long), cudaMemcpyHostToDevice );

    double* prob_device;
    cudaMalloc( (void**)&prob_device, n_edge[0]*sizeof(double));
    cudaMemcpy( prob_device, prob, n_edge[0]*sizeof(double), cudaMemcpyHostToDevice );

    long long* alias_device;
    cudaMalloc( (void**)&alias_device, n_edge[0]*sizeof(long long));
    cudaMemcpy( alias_device, alias, n_edge[0]*sizeof(long long), cudaMemcpyHostToDevice );

    long edge_count_actual = 0;
    long* edge_count_actual_device;
    cudaMalloc( (void**)&edge_count_actual_device, sizeof(long));
    cudaMemcpy( edge_count_actual_device, &edge_count_actual, sizeof(long), cudaMemcpyHostToDevice );

    long long* n_samples_device;
    cudaMalloc( (void**)&n_samples_device, sizeof(long long));
    cudaMemcpy( n_samples_device, n_samples, sizeof(long long), cudaMemcpyHostToDevice );

    int* edge_from_device;
    cudaMalloc( (void**)&edge_from_device, edge_ft_size*sizeof(int));
    cudaMemcpy( edge_from_device, edge_from, edge_ft_size*sizeof(int), cudaMemcpyHostToDevice );

    int* edge_to_device;
    cudaMalloc( (void**)&edge_to_device, edge_ft_size*sizeof(int));
    cudaMemcpy( edge_to_device, edge_to, edge_ft_size*sizeof(int), cudaMemcpyHostToDevice );

    long long* n_negatives_device;
    cudaMalloc( (void**)&n_negatives_device, sizeof(long long));
    cudaMemcpy( n_negatives_device, &n_negatives, sizeof(long long), cudaMemcpyHostToDevice );

    int* neg_table_device;
    cudaMalloc( (void**)&neg_table_device, neg_size*sizeof(int));
    cudaMemcpy( neg_table_device, neg_table, neg_size*sizeof(int), cudaMemcpyHostToDevice );

    long long* neg_size_device;
    cudaMalloc( (void**)&neg_size_device, sizeof(long long));
    cudaMemcpy( neg_size_device, &neg_size, sizeof(long long), cudaMemcpyHostToDevice );



    double *vis_d_device;
    cudaMalloc( (void**)&vis_d_device, n_vertices[0]*out_dim[0]*sizeof(double));
    cudaMemcpy( vis_d_device, vis_d, n_vertices[0]*out_dim[0]*sizeof(double), cudaMemcpyHostToDevice );


    //dim3 dimBlock( blocksize, 2);
    //dim3 dimGrid( 1, 1 );
    time_t start,end;
    time(&start);
    hello<<<600, 1000>>>(initial_alpha_device, out_dim_device, n_edge_device, vis_d_device, prob_device, alias_device, edge_count_actual_device, n_samples_device, edge_from_device, edge_to_device,n_negatives_device,neg_table_device, neg_size_device);



    cudaMemcpy( vis_d, vis_d_device, n_vertices[0]*out_dim[0]*sizeof(double), cudaMemcpyDeviceToHost );
    cudaFree(out_dim_device );
    cudaFree(initial_alpha_device );
    cudaFree(vis_d_device);

    std::cout<<vis_d[0]<<","<<vis_d[1]<<","<<vis_d[2]<<","<<vis_d[3]<<","<<vis_d[4]<<std::endl;
    time(&end);
    printf("GPU Training Time: %f\n",difftime(end,start));
}

