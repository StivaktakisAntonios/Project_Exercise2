#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <cmath>
#include <cstring>

#include "ivfpq.hpp"
#include "L2.hpp" 

using namespace std;

IVFPQ::IVFPQ(const Params& P) : P_Q(P) {
    if (P_Q.dim == 0)
        throw invalid_argument("IVFPQ: dim must be > 0");
    if (P_Q.kclusters == 0)
        throw invalid_argument("IVFPQ: kclusters must be > 0");
    if (P_Q.nprobe == 0)
        throw invalid_argument("IVFPQ: nprobe must be > 0");
    if (P_Q.M == 0)
        throw invalid_argument("IVFPQ: M must be > 0");
    if (P_Q.dim % P_Q.M != 0)
        throw invalid_argument("IVFPQ: dim must be divisible by M");
}


void IVFPQ::build(const double* data, size_t n){
    if (!data) 
        throw invalid_argument("IVFPQ::build: data is null");
    data_ = data;
    n_ = n;

     // Training (coarse + PQ)
     const double* X = data_;
     size_t nX = n_;

    // allocate IVF & PQ structures
    coarse_.assign(P_Q.kclusters * P_Q.dim, 0.0);
    inv_ids_.assign(P_Q.kclusters, {});
    inv_codes_.assign(P_Q.kclusters, {});
    
    pq_codebooks_.assign(P_Q.M * P_Q.Ksub * dim_sub(), 0.0);

    mt19937 rng(P_Q.seed);

    //Coarse k-means (full dim)
    run_kmeans_coarse(rng, X, nX);

    //PQ k-means per block (dim_sub)
    //For each block b, learn a codebook with Ksub codewords
    {
        vector<double> Cbk(P_Q.Ksub * dim_sub(), 0.0);
        for (size_t b = 0; b < P_Q.M; ++b){
            kmeans_pp_init_block(rng, X, nX, b, Cbk);
            run_kmeans_block(X, nX, b, Cbk);
            // Copy into the large buffer pq_codebooks_
            // Cbk[k*dim_sub + j] -> pq_codebooks_[(b*Ksub + k)*dim_sub + j]
            for (size_t k = 0; k < P_Q.Ksub; ++k){
                const double* src = &Cbk[k * dim_sub()];
                double* dst = const_cast<double*>(codeword_ptr(b, k));
                std::copy_n(src, dim_sub(), dst);
            }
        }
    }

    //Assign data vectors se IVF lists + PQ encode ===
    for (size_t i = 0; i < n_; ++i){
        const double* xi = data_ + i * P_Q.dim;

        //coarse assign
        int cid = assign_coarse(xi);
        if (cid < 0) 
            continue;

        // pq encode
        auto& ids = inv_ids_[(size_t)cid];
        auto& codes = inv_codes_[(size_t)cid];
            
        pq_encode_vector(xi, codes);   //pq_encode_vector appends M bytes for this vector
        ids.push_back((int)i);

    }
}

// ====================== Query APIs ======================

std::vector<std::pair<int,double>> IVFPQ::query_top(const double* q, size_t N) const{
    if (!data_) 
        throw logic_error("IVFPQ::query_top: index not built");
    if (N == 0) 
        return {};

    // select the nprobe nearest coarse centroids
    auto nearC = top_nprobe_centroids(q);

    // build LUTs for the query (squared L2 from q_block to codebook_b[k])
    // LUT is stored as: block-major, then k (size: M*Ksub)
    vector<double> LUT(P_Q.M * P_Q.Ksub, 0.0);
    build_LUTs_for_query(q, LUT);

    // scan the candidates with ADC (sum of M lookups)
    // keep top-N using a simple vector + partial sort (when needed)
    vector<pair<int,double>> res; 
    res.reserve(1024);

    for (auto [cid, _d2] : nearC){
        const auto& ids = inv_ids_[(size_t)cid];
        const auto& codes = inv_codes_[(size_t)cid];
        const size_t m = P_Q.M;

        for (size_t t = 0; t < ids.size(); ++t){
            const uint8_t* code_ptr = &codes[t * m];
            double d2 = adc_distance_sq_from_code(LUT, code_ptr);
            res.emplace_back(ids[t], d2);
        }
    }

    if (res.empty()) 
        return {};

    // keep the N smallest (by squared L2), then take sqrt
    if (res.size() > N){
        std::nth_element(res.begin(), res.begin() + (ptrdiff_t)N, res.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
        res.resize(N);
    }
    std::sort(res.begin(), res.end(),
              [](const auto& a, const auto& b){ return a.second < b.second; });

    for (auto& kv : res) 
        kv.second = std::sqrt(kv.second);
    return res;
}

std::vector<int> IVFPQ::query_range(const double* q, double R) const{
    if (!data_) 
        throw logic_error("IVFPQ::query_range: index not built");
    if (!(R > 0.0)) 
        return {};
    double R2 = R * R;

    auto nearC = top_nprobe_centroids(q);

    vector<double> LUT(P_Q.M * P_Q.Ksub, 0.0);
    build_LUTs_for_query(q, LUT);

    vector<int> out; out.reserve(1024);
    for (auto [cid, _d2] : nearC){
        const auto& ids = inv_ids_[(size_t)cid];
        const auto& codes = inv_codes_[(size_t)cid];
        const size_t m = P_Q.M;

        for (size_t t = 0; t < ids.size(); ++t){
            const uint8_t* code_ptr = &codes[t * m];
            double d2 = adc_distance_sq_from_code(LUT, code_ptr);
            if (d2 <= R2) out.push_back(ids[t]);
        }
    }
    return out;
}

// ====================== Coarse k-means (full dim) ======================
//
//static inline double l2_sq_full(const double* a, const double* b, size_t d){
//    double s=0.0; 
//    for(size_t i=0;i<d;++i){ 
//        double x=a[i]-b[i];
//        s += x*x;
//    }
//    return s;
//}

void IVFPQ::kmeans_pp_init_coarse(std::mt19937& rng, const double* X, std::size_t nX){
    // 1st centroid: random choice in [0, nX-1]
    uniform_int_distribution<size_t> U(0, nX-1);
    size_t first = U(rng);
    std::copy_n(X + first * P_Q.dim, P_Q.dim, &coarse_[0]);

    vector<double> dist2(nX, numeric_limits<double>::infinity());

    auto update_dist2 = [&](size_t cidx){
        const double* c = &coarse_[cidx * P_Q.dim];
        for (size_t i=0;i<nX;++i){
            const double* xi = X + i * P_Q.dim;
            double d2 = l2_double_squared(xi, c, P_Q.dim);
            if (d2 < dist2[i]) 
                dist2[i] = d2;
        }
    };
    update_dist2(0);

    for (size_t c=1;c<P_Q.kclusters;++c){
        double sum=0.0; 
        for(double v:dist2) 
            sum+=v;
        size_t pick;
        if (!(sum>0.0)){
            pick = U(rng);
        } else {
            uniform_real_distribution<double> Ur(0.0, sum);
            double r = Ur(rng), acc=0.0; pick=0;
            for (; pick<nX; ++pick){ 
                acc+=dist2[pick]; 
                if(acc>=r) 
                    break;
            }
            if (pick>=nX) 
                pick=nX-1;
        }
        std::copy_n(X + pick*P_Q.dim, P_Q.dim, &coarse_[c * P_Q.dim]);
        update_dist2(c);
    }
}

void IVFPQ::run_kmeans_coarse(std::mt19937& rng, const double* X, std::size_t nX){
    // coarse_ must be allocated before this is called
    kmeans_pp_init_coarse(rng, X, nX);

    vector<int> assign(nX, -1);
    for (size_t it=0; it<P_Q.max_iters; ++it){
        vector<double> sums(P_Q.kclusters * P_Q.dim, 0.0);
        vector<size_t> counts(P_Q.kclusters, 0);

        // assign
        for (size_t i=0;i<nX;++i){
            const double* xi = X + i*P_Q.dim;
            int best = -1; double bestd = numeric_limits<double>::infinity();
            for (size_t c=0;c<P_Q.kclusters;++c){
                const double* cj = &coarse_[c * P_Q.dim];
                double d2 = l2_double_squared(xi, cj, P_Q.dim);
                if (d2<bestd){ bestd=d2; best=(int)c; }
            }
            assign[i] = best;
            size_t base=(size_t)best*P_Q.dim;
            for (size_t j=0;j<P_Q.dim;++j) sums[base+j]+=xi[j];
            counts[(size_t)best]++;
        }

        // recompute + stop criterion
        double max_shift=0.0;
        for (size_t c=0;c<P_Q.kclusters;++c){
            if (counts[c]==0) 
                continue;
            double inv = 1.0/(double)counts[c];
            double* cj = &coarse_[c*P_Q.dim];
            double shift2 = 0.0;
            for (size_t j=0;j<P_Q.dim;++j){
                double newv = sums[c*P_Q.dim + j]*inv;
                double diff = cj[j]-newv;
                shift2 += diff*diff;
                cj[j]=newv;
            }
            if (shift2>max_shift) 
                max_shift=shift2;
        }
        if (max_shift<=P_Q.tol) 
            break;
    }
}

int IVFPQ::assign_coarse(const double* x) const{
    int best=-1; 
    double bestd=numeric_limits<double>::infinity();
    for (size_t c=0;c<P_Q.kclusters;++c){
        const double* cj=&coarse_[c*P_Q.dim];
        double d2 = l2_double_squared(x, cj, P_Q.dim);
        if (d2<bestd){ 
            bestd=d2; 
            best=(int)c; 
        }
    }
    return best;
}

std::vector<std::pair<int,double>> IVFPQ::top_nprobe_centroids(const double* q) const{
    vector<pair<int,double>> cand; 
    cand.reserve(P_Q.kclusters);
    for (size_t c=0;c<P_Q.kclusters;++c){
        const double* cj=&coarse_[c*P_Q.dim];
        cand.emplace_back((int)c, l2_double_squared(q, cj, P_Q.dim));
    }
    if (cand.size()>P_Q.nprobe){
        nth_element(cand.begin(), cand.begin()+(ptrdiff_t)P_Q.nprobe, cand.end(), [](const auto& a, const auto& b){return a.second<b.second;});
        cand.resize(P_Q.nprobe);
    }
    sort(cand.begin(), cand.end(), [](const auto& a, const auto& b){return a.second<b.second;});
    return cand;
}

// ====================== PQ k-means per block ======================

//static inline double l2_sq_block(const double* x, const double* y, size_t ds){
//    double s=0.0; 
    //for(size_t i=0;i<ds;++i){ 
        //double d=x[i]-y[i]; 
        //s+=d*d; } return s;
//}

// Direct access to block b of vector x: x_block[j] = x[b*ds + j]
static inline const double* block_ptr(const double* x, size_t b, size_t ds){
    return x + b*ds;
}

void IVFPQ::kmeans_pp_init_block(std::mt19937& rng, const double* X, std::size_t nX, std::size_t b, std::vector<double>& Cbk){
    // Cbk: Ksub * dim_sub
    const size_t ds = dim_sub();
    uniform_int_distribution<size_t> U(0, nX-1);
    size_t first = U(rng);
    // copy block b of X[first] into codeword 0
    std::copy_n(block_ptr(X + first*P_Q.dim, b, ds), ds, &Cbk[0]);

    vector<double> dist2(nX, numeric_limits<double>::infinity());

    auto update_dist2 = [&](size_t cidx){
        const double* c = &Cbk[cidx * ds];
        for (size_t i=0;i<nX;++i){
            const double* xi_block = block_ptr(X + i*P_Q.dim, b, ds);
            double d2 = l2_double_squared(xi_block, c, ds);
            if (d2 < dist2[i]) 
                dist2[i] = d2;
        }
    };
    update_dist2(0);

    for (size_t k=1;k<P_Q.Ksub;++k){
        double sum=0.0; 
        for(double v:dist2) 
            sum+=v;
        size_t pick;
        if (!(sum>0.0)){
            pick = U(rng);
        } else {
            uniform_real_distribution<double> Ur(0.0, sum);
            double r=Ur(rng), acc=0.0; pick=0;
            for (; pick<nX; ++pick){ 
                acc+=dist2[pick]; 
                if(acc>=r) 
                    break; 
            }
            if (pick>=nX) 
                pick=nX-1;
        }
        const double* src = block_ptr(X + pick*P_Q.dim, b, ds);
        std::copy_n(src, ds, &Cbk[k * ds]);
        update_dist2(k);
    }
}

void IVFPQ::run_kmeans_block(const double* X, std::size_t nX, std::size_t b, std::vector<double>& Cbk){
    const size_t ds = dim_sub();
    vector<int> assign(nX, -1);

    for (size_t it=0; it<P_Q.max_iters; ++it){
        vector<double> sums(P_Q.Ksub * ds, 0.0);
        vector<size_t> counts(P_Q.Ksub, 0);

        // assign
        for (size_t i=0;i<nX;++i){
            const double* xi_b = block_ptr(X + i*P_Q.dim, b, ds);
            int best=-1; 
            double bestd=numeric_limits<double>::infinity();
            for (size_t k=0;k<P_Q.Ksub;++k){
                const double* ck = &Cbk[k * ds];
                double d2 = l2_double_squared(xi_b, ck, ds);
                if (d2<bestd){
                    bestd=d2; 
                    best=(int)k; 
                }
            }
            assign[i]=best;
            const size_t base = (size_t)best * ds;
            for (size_t j=0;j<ds;++j) 
                sums[base+j]+=xi_b[j];
            counts[(size_t)best]++;
        }

        // recompute
        double max_shift=0.0;
        for (size_t k=0;k<P_Q.Ksub;++k){
            if (counts[k]==0) 
                continue;
            double inv = 1.0/(double)counts[k];
            double* ck = &Cbk[k * ds];
            double shift2=0.0;
            for (size_t j=0;j<ds;++j){
                double newv = sums[k*ds + j] * inv;
                double diff = ck[j]-newv;
                shift2 += diff*diff;
                ck[j]=newv;
            }
            if (shift2>max_shift) 
                max_shift=shift2;
        }
        if (max_shift<=P_Q.tol) 
            break;
    }
}

// ====================== PQ encode + LUT/ADC ======================

int IVFPQ::best_codeword_block(const double* x_block, const double* cbk) const{
    const size_t ds = dim_sub();
    int best=-1; double bestd=numeric_limits<double>::infinity();
    for (size_t k=0;k<P_Q.Ksub;++k){
        const double* ck = &cbk[k * ds];
        double d2 = l2_double_squared(x_block, ck, ds);
        if (d2<bestd){ 
            bestd=d2; 
            best=(int)k; 
        }
    }
    return best;
}

void IVFPQ::pq_encode_vector(const double* x, std::vector<uint8_t>& codes) const{
    // Append to the END of `codes` the M bytes encoding for vector x
    const size_t ds = dim_sub();
    size_t base = codes.size();
    codes.resize(base + P_Q.M);

    for (size_t b=0; b<P_Q.M; ++b){
        const double* x_b = block_ptr(x, b, ds);
        const double* cbk = codeword_ptr(b, 0); // base at (b,k=0); then offset by k*ds inside codeword_ptr
        int best = best_codeword_block(x_b, cbk);
        codes[base + b] = static_cast<uint8_t>(best); // Ksub <= 256
    }
}

void IVFPQ::build_LUTs_for_query(const double* q, std::vector<double>& LUT) const{
    // LUT size: M*Ksub. For each block b: LUT[b*Ksub + k] = L2^2(q_b, codebook_b[k])
    const size_t ds = dim_sub();
    for (size_t b=0; b<P_Q.M; ++b){
        const double* q_b = block_ptr(q, b, ds);
        for (size_t k=0; k<P_Q.Ksub; ++k){
            const double* ck = codeword_ptr(b, k);
            LUT[b*P_Q.Ksub + k] = l2_double_squared(q_b, ck, ds);
        }
    }
}

double IVFPQ::adc_distance_sq_from_code(const std::vector<double>& LUT, const uint8_t* code_ptr) const{
    // Sum M lookups: sum_b LUT[b*Ksub + code[b]]
    double s=0.0;
    for (size_t b=0; b<P_Q.M; ++b){
        uint8_t idx = code_ptr[b];
        s += LUT[b*P_Q.Ksub + (size_t)idx];
    }
    return s;
}