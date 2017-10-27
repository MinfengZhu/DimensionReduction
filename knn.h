#ifndef KNN_H
#define KNN_H

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include "ANNOY/annoylib.h"
#include "ANNOY/kissrandom.h"

using namespace std;

typedef float real;

class knn{
private:

public:
	knn();
    void contruct_knn();
    void knn_largevis();
	void run_annoy();
	void run_propagation();
    void knn_efanna();
    void test_accuracy();
    void compute_similarity();
};

#endif
