#include "efanna.hpp"
#include <iostream>
#include <fstream>
#include <ctime>
#include <malloc.h>
#include "LargeVis.h"

using namespace efanna;
using namespace std;

char infile[1000], outfile[1000];
long long if_embed = 1, out_dim = -1, n_samples = -1, n_threads = -1, n_negative = -1, n_neighbors = -1, n_trees = -1, n_propagation = -1;
int knn_trees = -1, epochs = -1;
int mlevel = -1, L = -1, checkK = -1, knn_k = -1, S = -1,build_trees = -1;
real alpha = -1, n_gamma = -1, perplexity = -1;

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char** argv){
  //if(argc!=11){cout<< argv[0] << " data_file save_graph_file trees level epoch L K kNN S" <<endl; exit(-1);}
	long long i;
	if ((i = ArgPos((char *) "-fea", argc, argv)) > 0) if_embed = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-input", argc, argv)) > 0) strcpy(infile, argv[i + 1]);
	if ((i = ArgPos((char *) "-output", argc, argv)) > 0) strcpy(outfile, argv[i + 1]);
	if ((i = ArgPos((char *) "-outdim", argc, argv)) > 0) out_dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-samples", argc, argv)) > 0) n_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) n_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-neg", argc, argv)) > 0) n_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-neigh", argc, argv)) > 0) n_neighbors = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-trees", argc, argv)) > 0) n_trees = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-prop", argc, argv)) > 0) n_propagation = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-gamma", argc, argv)) > 0) n_gamma = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-perp", argc, argv)) > 0) perplexity = atof(argv[i + 1]);

	if ((i = ArgPos((char *) "-knn_trees", argc, argv)) > 0) knn_trees = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-mlevel", argc, argv)) > 0) mlevel = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-epochs", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-L", argc, argv)) > 0) L = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-checkK", argc, argv)) > 0) checkK = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-knn_k", argc, argv)) > 0) knn_k = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-S", argc, argv)) > 0) S = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-build_trees", argc, argv)) > 0) build_trees = atoi(argv[i + 1]);

    n_neighbors = knn_k;


//	 strcpy(infile, "./data/twitter.200d.txt");
//	 strcpy(outfile, "./data/twitter.2d.txt");

//	strcpy(infile, "./data/sift_base.txt");
//	strcpy(outfile, "./data/sift_base.2d.txt");

	strcpy(infile, "./data/mnist_vec784D.txt");
	strcpy(outfile, "./data/mnist_vec2D_2.txt");

	LargeVis model;

	model.load_from_file(infile);
	model.run(out_dim, n_threads, n_samples, n_propagation, alpha, n_trees, n_negative, n_neighbors, n_gamma,
			  perplexity, knn_trees, mlevel, epochs, L, checkK, knn_k, S, build_trees);//n_neighbors
	model.save_gpu(outfile);


	//index.saveGraph(argv[2]);
	//index.saveTrees(argv[3]);
	return 0;
}
