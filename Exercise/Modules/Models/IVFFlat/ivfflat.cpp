#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cmath>

#include "ivfflat.hpp"
#include "L2.hpp"
#include "utils.hpp"

using namespace std;

IVFFlat::IVFFlat(const Params& P) : P_(P) {
    if (P_.dim == 0) 
        throw invalid_argument("IVFFlat: dim must be > 0");
    if (P_.kclusters == 0) 
        throw invalid_argument("IVFFlat: kclusters must be > 0");
    if (P_.nprobe == 0) 
        throw invalid_argument("IVFFlat: nprobe must be > 0");
}

void IVFFlat::build(const double* data, size_t n){
    if (!data) 
        throw invalid_argument("IVFFlat::build: data is null");
    data_ = data;
    n_ = n;

    if (P_.kclusters > n_) 
        const_cast<Params&>(P_).kclusters = n_;
    const_cast<Params&>(P_).nprobe = min(P_.nprobe, P_.kclusters);

    // allocate centroids & lists
    centroids_.assign(P_.kclusters * P_.dim, 0.0);
    lists_.assign(P_.kclusters, {});

    mt19937 rng(P_.seed);
    run_kmeans(rng);
}

vector<pair<int,double>> IVFFlat::query_top(const double* q, size_t N) const{
    if (!data_) 
        throw logic_error("IVFFlat::query_top: index not built");
    if (N == 0) 
        return {};

    //1) find the nprobe nearest centroids
    auto candsC = top_nprobe_centroids(q); // (centroid_id, L2)

    //2) gather candidate ids from the corresponding lists
    vector<int> candidates;
    size_t total = 0;
    for (size_t i = 0; i < candsC.size(); ++i)
        total += lists_[ (size_t)candsC[i].first ].size();
    candidates.reserve(total);
    for (auto [cid, _] : candsC) {
        const auto& L = lists_[(size_t)cid];
        candidates.insert(candidates.end(), L.begin(), L.end());
    }

    if (candidates.empty()) 
        return {};
    // 3) compute top-N over these candidates (use squared distances for speed, then sqrt at the end)
    auto top = knn_topN_cand(data_, P_.dim, q, N, candidates, &l2_double_squared);
    for (auto& kv : top) 
        kv.second = std::sqrt(kv.second);
    return top;
}

vector<int> IVFFlat::query_range(const double* q, double R) const{
    if (!data_) 
        throw logic_error("IVFFlat::query_range: index not built");
    if (!(R > 0.0)) 
        return {};
    // 1) find the nprobe nearest centroids
    auto candsC = top_nprobe_centroids(q);
    // 2) gather candidate ids from the corresponding lists
    vector<int> candidates;
    size_t total = 0;
    for (auto& pr : candsC) total += lists_[(size_t)pr.first].size();
    candidates.reserve(total);
    for (auto [cid, _] : candsC) {
        const auto& L = lists_[(size_t)cid];
        candidates.insert(candidates.end(), L.begin(), L.end());
    }
    if (candidates.empty()) 
        return {};
    // 3) range search on those candidates using R^2 with squared L2
    const double R2 = R * R;
    return range_search_cand(data_, P_.dim, q, R2, candidates, &l2_double_squared);
}

// ==================== K-MEANS ====================

void IVFFlat::run_kmeans(std::mt19937& rng){
    // 1) initialization
    kmeans_pp_init(rng);

    // temporary assignment array for Lloyd's iterations
    vector<int> assign(n_, -1);

    // 2) Lloyd iterations
    double max_shift = 0.0;
    for (size_t it = 0; it < P_.max_iters; ++it){
        // (a) assign
        // clear per-iteration accumulators (sums and counts) for clusters
        vector<double> sums(P_.kclusters * P_.dim, 0.0);
        vector<size_t> counts(P_.kclusters, 0);

        for (size_t i = 0; i < n_; ++i){
            const double* xi = data_ + i * P_.dim;

            // find nearest centroid
            int best_c = -1;
            double best_d2 = numeric_limits<double>::infinity();
            for (size_t c = 0; c < P_.kclusters; ++c){
                // l2_double_squared(xi, centroid_c)
                const double* cj = &centroids_[c * P_.dim];
                double d2 =  l2_double_squared(xi, cj, P_.dim);

                if (d2 < best_d2) { 
                    best_d2 = d2; 
                    best_c = (int)c; 
                }
            }
            assign[i] = best_c;

            // add to this cluster's sum
            const size_t base = (size_t)best_c * P_.dim;
            for (size_t j = 0; j < P_.dim; ++j)
                sums[base + j] += xi[j];
            counts[(size_t)best_c]++;
        }
        
        // (b) recompute centroids and track maximum shift
        max_shift = 0.0;
        bool had_empty = false;
        for (size_t c = 0; c < P_.kclusters; ++c){
            double* cj = &centroids_[c * P_.dim];
            if (counts[c] == 0) {
                // empty cluster: re-seed to a random datapoint
                uniform_int_distribution<size_t> U(0, n_ - 1);
                size_t pick = U(rng);
                std::copy_n(data_ + pick * P_.dim, P_.dim, cj);
                had_empty = true;
                continue; // skip shift computation
            }
            const double inv = 1.0 / static_cast<double>(counts[c]);
            double shift2 = 0.0;
            for (size_t j = 0; j < P_.dim; ++j){
                double newv = sums[c * P_.dim + j] * inv;
                double diff = cj[j] - newv;
                shift2 += diff * diff;
                cj[j] = newv;
            }
            if (shift2 > max_shift) max_shift = shift2;
        }
        // handle empty clusters by forcing another iteration
        if (had_empty) {
            // ensure at least one more iteration
            max_shift = std::max(max_shift, P_.tol * 2.0);
        }

            if (max_shift <= P_.tol) break;
        }

    // 3) build final inverted index
    rebuild_inverted_lists(assign);
}

// k-means++ init
void IVFFlat::kmeans_pp_init(std::mt19937& rng){
    // first centroid: uniform random choice
    uniform_int_distribution<size_t> U(0, n_ - 1);
    size_t first = U(rng);
    centroids_.assign(P_.kclusters * P_.dim, 0.0);
    std::copy_n(data_ + first * P_.dim, P_.dim, &centroids_[0]);

    // temporary D(x)^2 for every point
    vector<double> dist2(n_, numeric_limits<double>::infinity());

    // helper: update dist2 using the newest centroid
    auto update_dist2 = [&](size_t cidx){
        const double* c = &centroids_[cidx * P_.dim];
        for (size_t i = 0; i < n_; ++i){
            const double* xi = data_ + i * P_.dim;
            double d2 = 0.0;
            for (size_t j = 0; j < P_.dim; ++j) {
                double diff = xi[j] - c[j];
                d2 += diff * diff;
            }
            if (d2 < dist2[i]) dist2[i] = d2;
        }
    };

    update_dist2(0);

    // select the remaining k-1 centroids with probability ∝ D(x)^2
    for (size_t c = 1; c < P_.kclusters; ++c){
        // prefix sum over D(x)^2
        double sum = 0.0;
        for (double v : dist2) sum += v;
        if (!(sum > 0.0)) {
            // all identical → pick random
            size_t idx = U(rng);
            std::copy_n(data_ + idx * P_.dim, P_.dim, &centroids_[c * P_.dim]);
            update_dist2(c);
            continue;
        }

        uniform_real_distribution<double> Ur(0.0, sum);
        double r = Ur(rng);

        size_t pick = 0;
        double acc = 0.0;
        for (; pick < n_; ++pick){
            acc += dist2[pick];
            if (acc >= r) 
                break;
        }
        if (pick >= n_) 
            pick = n_ - 1;

        std::copy_n(data_ + pick * P_.dim, P_.dim, &centroids_[c * P_.dim]);
        update_dist2(c);
    }
}

void IVFFlat::rebuild_inverted_lists(const vector<int>& assign){
    lists_.assign(P_.kclusters, {});
    for (size_t i = 0; i < n_; ++i){
        int cid = assign[i];
        if (cid < 0) 
            continue;
        lists_[(size_t)cid].push_back((int)i);
    }
}
//MIPOS NA VALOUME L2 me tropopoihsh gia na kanoume ta xentroids?
double IVFFlat::centroid_l2_sq(size_t ci, const double* q) const{
    const double* c = &centroids_[ci * P_.dim];
    double d2 = 0.0;
    for (size_t j = 0; j < P_.dim; ++j) {
        double diff = q[j] - c[j];
        d2 += diff * diff;
    }
    return d2;
}

vector<pair<int,double>> IVFFlat::top_nprobe_centroids(const double* q) const{
    vector<pair<int,double>> v;
    v.reserve(P_.kclusters);
    for (size_t c = 0; c < P_.kclusters; ++c)
        v.emplace_back((int)c, centroid_l2_sq(c, q));

    if (v.size() > P_.nprobe) {
        nth_element(v.begin(), v.begin() + (ptrdiff_t)P_.nprobe, v.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
        v.resize(P_.nprobe);
    }
    sort(v.begin(), v.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
    // optionally sqrt the centroid distances
    return v;
}