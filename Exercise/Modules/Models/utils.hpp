#pragma once
#include <cstddef>
#include <utility>
#include <vector>


// We implement metrics and helpers in the .tpp so all algorithms can reuse them.

using Distance_function = double(*)(const double* a, const double* b, std::size_t d); // Typedef for a function pointer to plug in distance functions (e.g., from L2.cpp) without tight coupling.


// For the entire dataset
// NN (N=1)

std::pair<int,double>nn_search(const double* data, std::size_t n, std::size_t dim, const double* q, Distance_function dist);

// kNN for top neighbors
std::vector<std::pair<int, double>>knn_topN(const double* data, std::size_t n, std::size_t dim, const double* q, std::size_t N, Distance_function dist);


// Range over all points with distance <= R

std::vector<int>range_search(const double* data, std:: size_t n, std::size_t dim, const double* q, double R, Distance_function dist);



// Candidates from a specific list of IDs
std::vector<std::pair<int,double>>knn_topN_cand(const double* data, std::size_t dim, const double* q, std::size_t N, const std::vector<int>& candidates, Distance_function dist);


// Range for specific IDs
std::vector<int>range_search_cand(const double* data, std::size_t dim, const double* q, double R, const std::vector<int>& candidates, Distance_function dist);
#include "utils.tpp"