#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <deque>
#include <unordered_set>

#include "hypercube.hpp"
#include "L2.hpp"

using namespace std;

// -----------------  HypercubeHasher Implementation -----------------
HypercubeHasher::HypercubeHasher(int d, int dprime, float w, uint64_t seed)
: seed_base_(seed), d_(d), dprime_(dprime), w_(w),
  v_(dprime), t_(dprime)
{
    // Check parameters
    if (d_ <= 0) throw invalid_argument("d must be > 0");
    if (dprime_ <= 0 || dprime_ > 64) throw invalid_argument("d' must be in [1,64]");
    if (!(w_ > 0.f)) throw invalid_argument("w must be > 0");

    mt19937_64 rng(seed);                      // generator with fixed seed
    normal_distribution<float>   N(0.f, 1.f);  // for v_i
    uniform_real_distribution<float> U(0.f, w_); // for t_i

    // Initialize v_ and t_
    for (int i = 0; i < dprime_; ++i) {
        v_[i].resize(d_);
        for (int j = 0; j < d_; ++j)
            v_[i][j] = N(rng);    // v_i ~ N(0,1)^d
        t_[i] = U(rng);           // t_i ~ U[0,w)
    }
}

// Calculate bucket index for bit i
inline int64_t HypercubeHasher::bucketIndex(int i, const Vec& x) const {
    double proj = 0.0;
    // compute v_i · x
    for (int j = 0; j < d_; ++j)
        proj += static_cast<double>(v_[i][j]) * x[j];
    proj += t_[i];
    return static_cast<int64_t>( floor(proj / w_) );
}

// Pack bit into position pos
uint64_t HypercubeHasher::packBit(uint64_t acc, bool bit, int pos) noexcept {
    if (bit) acc |= (uint64_t{1} << static_cast<unsigned>(pos));
    return acc;
}

// Deterministic bit for (i, bucket) using splitmix64 hashing
// NEW implementation from deterministic to random
bool HypercubeHasher::bitOf(const Vec& x, int i) const {
    double proj = 0.0;
    for (int j = 0; j < d_; ++j)
        proj += static_cast<double>(v_[i][j]) * x[j];
    proj += t_[i];
    // Αν η προβολή είναι θετική, επιστρέφει bit=1
    return proj >= 0.0;
}

// Compute vertex code for x
// NEW implementation
uint64_t HypercubeHasher::vertexOf(const Vec& x) const {
    if (static_cast<int>(x.size()) != d_)
        throw invalid_argument("vertexOf: x.size() != d");
    uint64_t code = 0;
    for (int i = 0; i < dprime_; ++i) {
        const bool bit = bitOf(x, i);
        code = packBit(code, bit, i);
    }
    return code;
}

// -----------------  HypercubeIndex Implementation -----------------
HypercubeIndex::HypercubeIndex(const HypercubeHasher& hasher)
: H(hasher), data(nullptr), n_(0), d_(0)
{
    // empty constructor body
}

// Build the hypercube index
void HypercubeIndex::build(const vector <Vec>& X){
    data = &X;      // point to dataset
    n_ = X.size();  // number of points
    d_ = (n_ > 0 ? X[0].size() : 0); // dimension

    // check dimension consistency
    if (n_ > 0 && static_cast<int>(d_) != H.dim())
        throw invalid_argument("build: dataset dim != hasher dim");

    cube.clear();       // clear any existing data
    cube.reserve(n_);    // reserve slots for n_ points unique vertices (keys)

    // Insert points into the hypercube
    for (size_t i = 0; i < n_; ++i) {
        if (X[i].size() !=  static_cast<std::size_t>(d_)) throw std::invalid_argument("build: non-uniform vector dimension");   
        const Vec& x = (*data)[i];          // get point
        uint64_t vertex = H.vertexOf(x);    // compute vertex
        cube[vertex].push_back(i);          // insert point id
    }
}

// Compute statistics about the hypercube
HypercubeIndex::CubeStats HypercubeIndex::computeStats() const {
    CubeStats s;
    s.nonEmpty = cube.size();

    // Compute total inserts and max bucket size
    for (const auto& entry : cube) {
        const auto& ids = entry.second;
        s.totalInserts += ids.size();
        if (ids.size() > s.maxBucket) s.maxBucket = ids.size();
    }

    // Compute average bucket size
    if (s.nonEmpty > 0)
        s.avgBucket = static_cast<double>(s.totalInserts) / s.nonEmpty;

    return s;
}

// Compute probe order for given base vertex
vector<uint64_t> HypercubeIndex::probeOrder(uint64_t base, int maxProbes) const {
    vector<uint64_t> order;
    if (maxProbes <= 0) return order;
    order.reserve(maxProbes);           // reserve space

    unordered_set<uint64_t> vis;        // prevent duplicates
    vis.reserve(static_cast<size_t>(maxProbes * 2));    // rough estimate

    deque<pair<uint64_t, int>> q;   // (vertex code, hamming distance)
    q.emplace_back(base, 0);        // start from base
    vis.insert(base);
    order.push_back(base);
    if ((int)order.size() >= maxProbes) return order;

    const int B = H.bits(); // d' bits

    // BFS-like approach to generate neighbors by flipping bits
    while (!q.empty() && (int)order.size() < maxProbes) {
        auto [code, dist] = q.front();
        q.pop_front();
        if (dist >= B) continue; // no more bits to flip

        for (int bit = 0; bit < B && (int)order.size() < maxProbes; ++bit) {
            uint64_t nei = code ^ (uint64_t{1} << static_cast<unsigned>(bit)); // flip bit
            if (!vis.insert(nei).second) continue; // already visited
            order.push_back(nei);
            q.emplace_back(nei, dist + 1);
        }
    }
    return order;
}

// Execute a query on the hypercube
vector<HypercubeIndex::Candidate>
HypercubeIndex::query(const Vec& q, const QueryParams& p,
                      vector<size_t>* rNeighbors,
                      size_t* examined) const
{
    if (static_cast<int>(q.size()) != H.dim())
        throw invalid_argument("query: q.size() != hasher dim");
    if (examined) *examined = 0;
    if (rNeighbors) rNeighbors->clear();

    const int N = max(1, p.N);
    const int M = max(1, p.M);
    const int PROBES = max(1, p.probes);

    const uint64_t base = H.vertexOf(q);
    const auto order = probeOrder(base, PROBES);

    vector<Candidate> cands;
    cands.reserve(min<size_t>(n_, static_cast<size_t>(M)));

    unordered_set<size_t> seen;                 // prevent duplicate examinations
    seen.reserve(static_cast<size_t>(M * 2));   // rough estimate

    int examinedDistinct = 0; 

    // Explore vertices in probe order
    for (uint64_t vtx : order) {
        auto it = cube.find(vtx);
        if (it == cube.end()) continue;

        const auto& ids = it->second;
        for (size_t id : ids) {
            if (examinedDistinct >= M) break;      // reached max distinct examined
            if (!seen.insert(id).second) continue; // skip duplicates
            
            // Compute distance
            const double dist = static_cast<double>(
                l2_float(q.data(), (*data)[id].data(), static_cast<size_t>(d_))
            );

            // Record range neighbor if applicable
            if (rNeighbors && p.R > 0.0 && dist <= p.R)
                rNeighbors->push_back(id);

            cands.push_back({id, dist});
            ++examinedDistinct;     // count distinct examined
            if (examined) (*examined) = examinedDistinct;

            if (examinedDistinct >= M) break;
        }
        if (examinedDistinct >= M) break;
    }

    // Select top-N candidates
    if ((int)cands.size() > N) {
        nth_element(cands.begin(), cands.begin() + N, cands.end(),
                         [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
        cands.resize(N);
    }
    sort(cands.begin(), cands.end(),
              [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });

    return cands;
}

// -------------------- Hypercube Implementation -----------------
//Constructor
Hypercube::Hypercube(const Params& P)
: P_(P),
  hasher_(P_.d, P_.dprime, static_cast<float>(P_.w), P_.seed),
  index_(hasher_)
{
    if (P_.dprime <= 0 || P_.dprime > 64)
        throw invalid_argument("Hypercube::Params.dprime must be in [1,64]");
}

// Build the hypercube index
void Hypercube::build(const vector<Vec>& X) {
    index_.build(X);
}

// Top-N approximate nearest neighbors
vector<pair<int,double>>
Hypercube::query_top(const Vec& q, int N) const {
    HypercubeIndex::QueryParams qp;
    qp.N = max(1, N);
    qp.R = 0.0;                 // not used in top-N
    qp.M = max(1, P_.M);
    qp.probes = max(1, P_.probes);
    
    size_t examined = 0;

    auto cand = index_.query(q, qp, nullptr, &examined);
    vector<pair<int,double>> out;
    out.reserve(cand.size());
    for (const auto& c : cand)
        out.emplace_back(static_cast<int>(c.id), c.dist);
    return out;
}

// Range query: return unique ids with L2 <= R
vector<int>
Hypercube::query_range(const Vec& q, double R) const {
    HypercubeIndex::QueryParams qp;
    qp.N = 1;                   // not used in range because we want all in range
    qp.R = R;
    qp.M = max(1, P_.M);
    qp.probes = max(1, P_.probes);

    vector<size_t> rids;
    index_.query(q, qp, &rids, nullptr);

    // Convert size_t ids to int ids
    vector<int> out;
    out.reserve(rids.size());
    for (auto id : rids)
        out.push_back(static_cast<int>(id));
    return out;
}