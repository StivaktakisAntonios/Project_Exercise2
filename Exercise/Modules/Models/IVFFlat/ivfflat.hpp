#pragma once
#include <vector>
#include <cstddef>
#include <random>
#include <utility>

class IVFFlat {
public:
    struct Params {
        std::size_t dim = 0; // dimensionality
        std::size_t kclusters = 50; // number of clusters (lists)
        std::size_t nprobe = 5; // how many nearest centers to check at query time
        unsigned seed = 1; // RNG
        std::size_t max_iters = 20; // k-means Lloyd iterations
        double tol = 1e-4; // early stop threshold (centroid L2 shift)
    };

    explicit IVFFlat(const Params& P);

    // Expose the dataset to the index (row-major: n*dim). Does not copy the data.
    void build(const double* data, std::size_t n);

    // Top-N from nprobe inverted lists (returns (id, L2))
    std::vector<std::pair<int,double>> query_top(const double* q, std::size_t N) const;

    // Range search (ids with L2 <= R) within nprobe inverted lists
    std::vector<int> query_range(const double* q, double R) const;

    // (προαιρετικό) για debugging/στατιστικά
    std::size_t num_lists() const { return lists_.size(); }
    std::size_t list_size(std::size_t i) const { return lists_[i].size(); }

private:
    Params P_;
    const double* data_ = nullptr; // not owned by the class
    std::size_t n_ = 0;

    // centroids: k * dim
    std::vector<double> centroids_; // row-major: c_i at &centroids_[i*dim]
    // inverted lists: for each cluster, the ids assigned to it
    std::vector<std::vector<int>> lists_;

    // --- internal helpers ---
    // k-means++ centroid initialization
    void kmeans_pp_init(std::mt19937& rng);
    // one Lloyd iteration: assign + recompute
    // returns max centroid L2 shift
    double lloyd_iteration();
    // run k-means
    void run_kmeans(std::mt19937& rng);

    // build inverted index from final assignments
    void rebuild_inverted_lists(const std::vector<int>& assign);

    // find top-nprobe centroids (returns pairs (centroid_id, L2))
    std::vector<std::pair<int,double>> top_nprobe_centroids(const double* q) const;

    // L2^2 between q and centroid i
    double centroid_l2_sq(std::size_t ci, const double* q) const;
};