#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

// Multidimensional vector type
using Vec = std::vector<float>;


// === HypercubeHasher ===
class HypercubeHasher {
public:
    // d: dimension of input points, dprime: number of bits (<=64), w: window size, seed: RNG seed
    HypercubeHasher(int d, int dprime, float w, uint64_t seed = 1);

    // Compute vertex code for input point x
    uint64_t vertexOf(const Vec& x) const;

    // Getters
    int  dim()     const noexcept { return d_; }
    int  bits()    const noexcept { return dprime_; }
    float cellW()  const noexcept { return w_; }

private:
    // Deterministic 64-bit mixer (splitmix64)
    static inline uint64_t splitmix64(uint64_t z) noexcept {
        z += 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    uint64_t seed_base_;  // base seed for deterministic bit mapping
    int   d_;
    int   dprime_;
    float w_;

    // Random projection vectors and offsets
    std::vector<Vec>   v_;   // v_i ∈ R^d
    std::vector<float> t_;   // t_i ∈ [0,w)

    // floor((v_i · x + t_i) / w)
    int64_t bucketIndex(int i, const Vec& x) const;

    // Pack bit into position pos
    static uint64_t packBit(uint64_t acc, bool bit, int pos) noexcept;

    // Get bit for bucket i
    bool bitOf(const Vec& x, int i) const;  // NEW param x

};

// === HypercubeIndex ===
class HypercubeIndex {
public:
    // Constructor with hasher
    HypercubeIndex(const HypercubeHasher& hasher);  
    void build(const std::vector<Vec>& X); 

    // Struct for cube statistics
    struct CubeStats {
        std::size_t nonEmpty = 0;
        std::size_t totalInserts = 0;
        std::size_t maxBucket = 0;
        double avgBucket = 0.0;
    };

    CubeStats computeStats() const;  // Compute statistics about the cube

    // Query parameters
    struct QueryParams {
        int N{1};               // top-N
        double R{0.0};          // range radius
        int M{10};              // max examined distinct candidates
        int probes{2};          // max probes
    };

    // Struct for candidate
    struct Candidate {
        std::size_t id;    // point id
        double dist;        // distance
    };

    // Execute a query on the hypercube
    std::vector<Candidate> 
    query(const Vec& q, const QueryParams& p, 
        std::vector<std::size_t>* rNeighbors = nullptr,
        std::size_t* examined = nullptr) const;


private:
    const HypercubeHasher& H;       // hasher reference
    std::unordered_map<uint64_t, std::vector<size_t>> cube;  // the hypercube
    const std::vector<Vec>* data;            // pointer to dataset

    size_t n_;                          // number of points
    int d_;                             // dimension

    std::vector<uint64_t> probeOrder(uint64_t base, int maxProbes) const;   // probe order
};


// === Hypercube ===
class Hypercube {
public:
    struct Params {
        int d = 0;          // dimension of input points
        int dprime = 14;    // bits (<=64)
        double w = 4.0;     // window
        int M = 10;         // max examined candidates
        int probes = 2;     // max probes
        uint64_t seed = 1;  // RNG
    };

    explicit Hypercube(const Params& P);

    // Build the hypercube index
    void build(const std::vector<Vec>& X);

    // Top-N approximate nearest neighbors
    std::vector<std::pair<int,double>>
    query_top(const Vec& q, int N) const;

    // Range query: return unique ids with L2 <= R
    std::vector<int>
    query_range(const Vec& q, double R) const;

private:
    Params P_;
    HypercubeHasher hasher_;
    HypercubeIndex  index_;
};