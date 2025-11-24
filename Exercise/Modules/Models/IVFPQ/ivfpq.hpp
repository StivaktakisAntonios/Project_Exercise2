/* Inverted File + Product Quantization */
// ivfpq.hpp
#pragma once
#include <vector>
#include <cstddef>
#include <random>
#include <utility>
#include <cstdint>
#include <stdexcept>

class IVFPQ {
public:
    struct Params {
        std::size_t dim = 0; // dimensionality d
        std::size_t kclusters = 50; // number of coarse centroids (inverted lists)
        std::size_t nprobe = 5; // how many nearest coarse centroids to scan at query time
        std::size_t M = 16; // number of PQ blocks (d must be divisible by M)
        std::size_t Ksub = 256; // codewords per block (8-bit index when <=256)
        unsigned seed = 1; // RNG
        std::size_t max_iters = 20; // k-means Lloyd iters
        double tol = 1e-4; // early stop threshold (L2 metablh centroid)W
    };

    explicit IVFPQ(const Params& P);

    // Index the dataset (row-major: n*dim). Does not copy the data.
    void build(const double* data, std::size_t n);

    // Top-N search with ADC over nprobe lists — returns sorted (id, L2)
    std::vector<std::pair<int,double>> query_top(const double* q, std::size_t N) const;

    // Range search with ADC (ids with L2 <= R) over nprobe lists
    std::vector<int> query_range(const double* q, double R) const;

    // (optional) stats
    std::size_t num_lists() const { 
        return inv_ids_.size(); 
    }
    std::size_t list_size(std::size_t i) const { 
        return inv_ids_[i].size(); 
    }

private:
    Params P_Q;

    // Pointers to dataset/training (not owned by the class)
    const double* data_ = nullptr;
    std::size_t n_ = 0;

    const double* train_data_ = nullptr;
    std::size_t tn_ = 0;

    // --- IVF (coarse) ---
    // Coarse centroids: kclusters * dim, row-major
    std::vector<double> coarse_;                // &coarse_[c*dim] is centroid c
    // Inverted lists: for each list keep ids and PQ codes (packed: M bytes/vec when Ksub<=25
    std::vector<std::vector<int>> inv_ids_;
    std::vector<std::vector<uint8_t>> inv_codes_;  // size == inv_ids_[list].size() * M

    // --- PQ codebooks ---
    // For each block b in [0..M-1], a codebook with Ksub codewords; each codeword has dim_sub dimensions
    // Storage: M * Ksub * dim_sub (row-major over (b,k))
    std::vector<double> pq_codebooks_; // pointer to (b,k): &pq_codebooks_[(b*Ksub + k)*dim_sub]*dim_sub]

    // --- Helpers ---
    std::size_t dim_sub() const { 
        if (P_Q.M == 0) return 0;
        return P_Q.dim / P_Q.M; 
    }

    // === Coarse k-means (full dim) ===
    void kmeans_pp_init_coarse(std::mt19937& rng, const double* X, std::size_t nX);
    void run_kmeans_coarse(std::mt19937& rng, const double* X, std::size_t nX);

    //Assign a vector to the nearest coarse centroid (full dim)
    int assign_coarse(const double* x) const;

    // Select top-nprobe coarse centroids for a query (returns (cid, L2^2))
    std::vector<std::pair<int,double>> top_nprobe_centroids(const double* q) const;

    // === PQ k-means (per block, dim_sub) ===
    void kmeans_pp_init_block(std::mt19937& rng, const double* X, std::size_t nX, std::size_t b, std::vector<double>& Cbk);
    void run_kmeans_block(const double* X, std::size_t nX, std::size_t b, std::vector<double>& Cbk);

    // Encode a vector into PQ codes (M bytes when Ksub<=256)
    void pq_encode_vector(const double* x, std::vector<uint8_t>& out_codes) const;
    //Helper: best codeword index for block b
    int best_codeword_block(const double* x_block, const double* cbk) const;

    //Build LUTs for the query: LUT[b][k] = L2^2(q_block_b, codebook_b[k])
    void build_LUTs_for_query(const double* q, std::vector<double>& LUT) const;
    //Compute ADC distance (L2^2) from M lookups for a packed code pointer
    double adc_distance_sq_from_code(const std::vector<double>& LUT, const uint8_t* code_ptr) const;

    // Retrieve pointer to codebook (b,k)
    const double* codeword_ptr(std::size_t b, std::size_t k) const 
    {
        std::size_t ds = dim_sub();
        return &pq_codebooks_[ (b * P_Q.Ksub + k) * ds ];
    }
};