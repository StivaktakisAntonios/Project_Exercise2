// Locality-Sensitive Hashing (LSH) for Euclidean (L2) metric
// -----------------------------------------------------------
// Goal: quickly find approximate nearest neighbours.
// Idea: hash each vector x into buckets using k independent L2-hash functions.
// We build L independent hash tables; for a query we only scan the
// candidates that land in the same buckets (not the whole dataset).
#include <random>
#include <cmath>
#include <cassert> //for aassert check rules
#include <stdexcept>
#include <unordered_set>
#include <algorithm>

#include "utils.hpp"
#include "LSH.hpp"
#include "L2.hpp"

using namespace std;
// ---------------------------- L2Hash ----------------------------
// Hash function for L2 (p-stable LSH):
// h(x) = floor( (a · x + b) / w )
// where:
// a ~ N(0,1)^d (Gaussian random projection)
// b ~ Uniform[0,w) (random shift)
// w > 0 (bucket width)


// Random constructor: samples a and b from the distributions above.
L2Hash::L2Hash(size_t dim, double w, mt19937& rng){
  assert(w>0.0);
  this->dim = dim;
  this->w = w;
  this->inv_w = 1.0 / w; // store 1/w to avoid repeated divisions
  a.resize(dim); //vector
  
  // Distributions for a and b
  normal_distribution<double> normal(0.0, 1.0); //for a
  uniform_real_distribution<double> uni(0.0, w); //for b
  
  // Sample a ~ N(0,1)^dim
  for(size_t i = 0; i < dim; i++){
    a[i] = normal(rng);
  }

  // Sample b ~ U[0, w)
  b = uni(rng);
}

// Deterministic constructor: use provided a, b, w.
// We clamp b into [0, w) to be safe and validate inputs.
L2Hash::L2Hash(const std::vector<double>& a, double b, double w)
{
  if (w <= 0.0) {
    throw std::invalid_argument("L2Hash: w must be > 0");
  }
  if (a.empty()) {
    throw std::invalid_argument("L2Hash: a must be non-empty");
  }
  this->dim = a.size();
  this->w = w;
  this->inv_w = 1.0 / w;
  this->a = a; 
  // Normalize b into [0, w)
  double bb = b;
  if (!std::isfinite(bb)) {
    bb = 0.0; // guard against NaN/Inf
  }
  // Wrap into [0, w) (two fmods handle negatives safely)
  bb = std::fmod(std::fmod(bb, this->w) + this->w, this->w);
  this->b = bb;
}


// Compute h(x) = floor((a·x + b)/w).
// Throws if dimension mismatches or x is null.
long long L2Hash::hash_function(const double* x, size_t d) const{
  double t;
  double sum = 0.0;
  long long bucket;
  if (d != dim){
    throw std::invalid_argument("L2Hash::hash_function: wrong dimension");
    //not using 0 value because may is valid price in a bucket
  }
  if (x == nullptr){
    throw std::invalid_argument("L2Hash::hash_function: null pointer");
  }
  
  for (size_t i = 0; i < d; i++){
    sum += a[i] * x[i]; //a einai i provoli sto N(0,1)
  }
  sum +=b;
  t = sum * inv_w;
  bucket = static_cast<long long>(std::floor(t));
  return bucket;
}
// --------------------------- Signature -------------------------
// A Signature holds k independent L2Hash functions and can:
// - compute the k-dimensional signature h(x) = [h1(x),...,hk(x)]
// - turn that signature into a stable string key like "12|-7|7|0"

Signature::Signature(size_t k, size_t dim, double w, mt19937& rng){
  this->k = k;
  this->dim = dim;
  this->w = w;

  assert(k>0);
  assert(dim>0);
  assert(w>0.0);

  hashes.reserve(k); // allocate once
  
  for(size_t i = 0; i < k; i++){
    hashes.emplace_back(dim, w, rng); // emplace a new L2Hash with its own random a,b
  }

}

// Compute signature vector: [h1(x), ..., hk(x)]
std::vector<long long> Signature::signature_function(const double* x, size_t d) const{
  if(x == nullptr || d != dim){
    throw std::invalid_argument
    ("Signature::signature_function invalid value");
  }  
    vector<long long> sig;
    sig.reserve(k); //allocate memory
    
    for(size_t i = 0; i < k; i++){
      long long h = hashes[i].hash_function(x, d); //compute hashes of x.  
      sig.push_back(h); //add result in the back
    }

  return sig;
}

// Build a stable string key like "h1|h2|...|hk" for use in hash tables.
string Signature::make_key(const vector<long long>& sig) const{
  assert(sig.size() == k);
  string key;
  key.reserve(k * 12); // rough guess to minimize reallocations
  for(size_t i = 0; i < k; i++){
    key += std::to_string(sig[i]);
    if(i+1 < k){
      key.push_back('|'); //add | to the buckets
    }
  }
return key;
}

// Convenience: compute signature and directly return its key.
string Signature::key_for(const double* x, size_t d) const{
  assert(x != nullptr && d == dim);
  vector <long long> vec;
  vec = signature_function(x, d);
  string key;
  key = make_key(vec);
  return key;
}



// ---------------------------- LSHTable -------------------------
// One LSH table = one Signature (k hashes) + buckets map.
// `buckets` maps key -> list of point IDs that landed in that bucket.
LSHTable::LSHTable(size_t k, size_t dim, double w, mt19937& rng): g(k, dim, w, rng), dim(dim){
  
}
// Insert a point (id, x) into the table.
void LSHTable::add_point(int id, const double* x, size_t d){
  //assert(x != nullptr && d == dim);
  if( x== nullptr || d != dim){
    throw invalid_argument("LSHTable::add_point invalid ptr or dimension");
  }
  string key;
  key = g.key_for(x, d);
  buckets[key].push_back(id);
}

// Return the candidate IDs for query q: all IDs that share the same bucket key.
vector<int> LSHTable::candidates(const double* q, size_t d)const {
  if (q == nullptr || d != dim){
    throw invalid_argument("LSHTable::candidates Invalid ptr or dimension");
  }
  string key;
  key = g.key_for(q,d);
  
  auto it = buckets.find(key);
  //amnazito buctes
  
  if(it == buckets.end()){
      return {};
  }
  else
  {
    return it->second;
  }

}

// ----------------------------- LSHIndex ------------------------
// An index is a collection of L independent LSH tables.
// Build: hash every data point into every table.
// Query: union the candidates from all tables, then compute real distances
// only for those candidates.

LSHIndex::LSHIndex(size_t dim, size_t k, size_t L, double w, mt19937& rng):dim(dim), L(L), w(w), k(k){
  if(dim == 0 || k == 0 || L == 0){
    throw invalid_argument("LSHIndex: dim, k, L must be >0");
  }
  lsh_tables.reserve(L);//desmevo xoro 
  for(size_t i =0; i < L; i++){
    lsh_tables.emplace_back(k, dim, w, rng);
  }
}

// Build the index on an n×dim row-major array data
void LSHIndex::build(const double* data, size_t n){
  if(!data) throw invalid_argument("LSHIndex::build data is null");
  data_ptr = data;
  n_points = n;

  // Insert every point into every table.
  for(size_t i = 0; i < n_points; ++i){
    
    const double* row = data_ptr + i * dim;

    for (auto& tbl : lsh_tables){

      tbl.add_point(static_cast<int>(i), row, dim);
    }
  }
}


// Query top-N approximate neighbours.
// Steps:
// 1) Collect candidates from all L tables (use a set to deduplicate).
// 2) Compute actual distances only for those candidates.
// 3) Keep the best N.
vector<pair<int, double>> LSHIndex::query_top(const double* q, size_t d, size_t N){
  if(d != dim) throw invalid_argument("LSIndex::query_top: query dimensinality");
  if (!data_ptr) throw std::logic_error("LSHIndex::query_top: index not built");
  if(N == 0){
    return {};
  }
  // Union of candidates from all tables (deduplicated).
  unordered_set<int> cand;
  cand.reserve(16 * L);
  for(const auto& tbl : lsh_tables){
    auto ids = tbl.candidates(q, d);
    cand.insert(ids.begin(), ids.end());
  }
  if(cand.empty()){
    return {};
  }

  // Move to a flat vector for distance computations.
  vector<int> cands;
  cands.reserve(cand.size());
  for(int id : cand){
    cands.emplace_back(id);
  }


  // Safety check: ensure IDs are in range.
  for (int id : cands) {
    if (id < 0 || (std::size_t)id >= n_points) {
      throw std::runtime_error("candidate id out of range: " + std::to_string(id));
    }
  }

  // Use squared L2 for speed, then sqrt before returning.
  auto top = knn_topN_cand(data_ptr, dim, q, N, cands, &l2_double_squared);
  for (auto& kv : top) kv.second = std::sqrt(kv.second);

  return top;
}

// Range search: return IDs with distance <= R (approximate).
vector<int> LSHIndex::query_range(const double* q, size_t d, double R){
  if(d != dim){
    throw invalid_argument("LSHIndex::query_range: query dimension problem");
  }
  if(R < 0){
    return {};
  }
  //require_built(data_ptr);
  if (!data_ptr) throw std::logic_error("LSHIndex::query_range: index not built");
  unordered_set<int> cand;
  cand.reserve(16 * L);
  for(const auto& tbl : lsh_tables){
    auto ids =tbl.candidates(q, d);
    cand.insert(ids.begin(), ids.end());
  }


  vector<int> cands;
  cands.reserve(cand.size());
  for(int id : cand){
    cands.emplace_back(id);
  }
  // Compare using squared L2 with R^2 (avoids sqrt per candidate).
  const double R2 = R * R;
  return range_search_cand(data_ptr, dim, q, R2, cands, &l2_double_squared);
}