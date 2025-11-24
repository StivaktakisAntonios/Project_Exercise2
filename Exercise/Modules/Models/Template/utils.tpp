#pragma once 
#include <algorithm>
#include <limits>
#include "utils.hpp"

// NN for N=1
inline std::pair<int,double>nn_search(const double* data, std::size_t n, std::size_t dim, const double* q, Distance_function dist){
  int best_id = -1; // variable that will hold the nearest neighbor index
  double best_d = std::numeric_limits<double>::infinity();
  for (std::size_t i=0; i<n; ++i){
    double d = dist(q, (data + i*dim), dim); // compute distance using the provided metric (e.g., L2)
    if (d < best_d){
      best_d = d;
      best_id = (int) i;
    }
  }
  return {best_id, best_d};// return the pair with best id and distance
}

// kNN for the top neighbors
// The dataset `data` has length n*dim; the i-th vector starts at data + i*dim.
// `q` is the query of length dim, `dist` is a function pointer to the metric (e.g., L2).
// Returns a vector of (id, distance) pairs.
inline std::vector<std::pair<int,double>>knn_topN(const double* data, std::size_t n, std::size_t dim, const double* q, std::size_t N, Distance_function dist){
  std::vector<std::pair<int,double>> res;


  if(N == 0 || n ==0) 
    return res; // quick check



  res.reserve(n); // reserve space
  for(std::size_t i=0; i<n; ++i){
    res.emplace_back((int)i, dist(q, data+i*dim, dim)); // compute distance for each point
  }

  if(res.size() > N){
    std::nth_element(res.begin(), res.begin()+static_cast<std::ptrdiff_t>(N), res.end(), [](const auto& a, const auto& b){return a.second < b.second;});
    res.resize(N);  
  }
  // sort the top-N in ascending distance and return
  std::sort(res.begin(), res.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
  return res;
}


//range search
inline std::vector<int>range_search(const double* data, std::size_t n, std::size_t dim, const double* q, double R, Distance_function dist){
  std::vector<int> out;
  out.reserve(64);
  for(std::size_t i=0; i<n; ++i){
    if(dist(q, data+i*dim, dim) <=R){ // if within radius R
      out.push_back((int)i); // push index (cast to int)
    }
  }
  return out;
}



// Specific candidates by IDs
inline std::vector<std::pair<int,double>>knn_topN_cand(const double* data, std::size_t dim, const double* q, std::size_t N, const std::vector<int>& candidates, Distance_function dist){
  std::vector<std::pair<int,double>> res;
  if(N == 0 || candidates.empty()){
    return res;
  }
  res.reserve(candidates.size());// Specific candidates by IDs
  for(int id : candidates){
    const double* row = data+(std::size_t)id*dim;
    res.emplace_back(id, dist(q, row, dim));
  }
  if(res.size()> N){
    std::nth_element(res.begin(), res.begin()+static_cast<std::ptrdiff_t>(N), res.end(), [](const auto& a, const auto& b){return a.second < b.second;});
    res.resize(N);
  }
  std::sort(res.begin(), res.end(),[](const auto& a, const auto& b){ return a.second < b.second; });
  return res;
  
}


// Range search on specific IDs
inline std::vector<int>range_search_cand(const double* data, std::size_t dim, const double* q, double R, const std::vector<int>& candidates, Distance_function dist){
  std::vector<int> out; // result container
  out.reserve(candidates.size()); // reserve according to candidates
  for (int id : candidates){ 
    const double* row = data + (std::size_t)id * dim;
    if (dist(q, row, dim) <= R){
      out.push_back(id);
    } 
  }
  return out;
}