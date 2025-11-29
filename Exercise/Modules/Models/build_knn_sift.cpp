#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <algorithm>   // για std::min

#include "Template/data_io.hpp"
#include "IVFFlat/ivfflat.hpp"

// Μικρό helper για usage
static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " -d <input.fvecs> -k <knn> -o <output.bin>\n";
}

int main(int argc, char** argv) {
    std::string data_path;
    std::string out_path;
    int k = 15;

    // --- απλό parsing των ορισμάτων ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "-k" && i + 1 < argc) {
            k = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (data_path.empty() || out_path.empty()) {
        usage(argv[0]);
        return 1;
    }
    if (k <= 0) k = 1;

    try {
        // 1. Φόρτωση SIFT από 1ης εργασίας
        //std::vector<double> data_vec;
        //std::size_t n = 0, dim = 0;
        //load_sift_fvecs(data_path, data_vec, n, dim);
        //const double* data = data_vec.data();
//
        //std::cerr << "Loaded SIFT dataset: n=" << n
        //          << ", dim=" << dim << std::endl;

        std::vector<double> data_vec;
        std::size_t n = 0, dim = 0;
        load_sift_fvecs(data_path, data_vec, n, dim);
        const double* data = data_vec.data();

        std::cerr << "Loaded SIFT dataset: n=" << n
                  << ", dim=" << dim << std::endl;

        // Χρησιμοποιούμε μόνο τα πρώτα 200000 σημεία για τον k-NN γράφο
        std::size_t n_used = std::min<std::size_t>(n, 200000);
        std::cerr << "Using only first " << n_used
                  << " points to build k-NN graph." << std::endl;


        //if ((std::size_t)k >= n) {
        //    k = static_cast<int>(n) - 1;
        //}

        if ((std::size_t)k >= n_used) {
            k = static_cast<int>(n_used) - 1;
        }


        // 2. Params για IVFFlat (μπορείς να τα ρυθμίσεις όπως στην 1η εργασία)
        IVFFlat::Params P;
        P.dim       = dim;
        P.kclusters = 1024; //4096;     // π.χ. 4096 κέντρα
        P.nprobe    = 16; //32;       // πόσα κέντρα θα σκανάρουμε στο query
        P.max_iters = 10; //20;       // k-means iters
        P.tol       = 1e-3; //1e-4;     // tolerance
        P.seed      = 1;        // seed

        IVFFlat index(P);
        std::cerr << "Building IVFFlat index..." << std::endl;
        index.build(data, n);
        std::cerr << "Index built." << std::endl;

        // 3. Υπολογισμός approximate k-NN για κάθε σημείο
        //std::vector<std::int32_t> all_ids;
        //all_ids.resize(n * static_cast<std::size_t>(k));

        std::vector<std::int32_t> all_ids;
        all_ids.resize(n_used * static_cast<std::size_t>(k));


        //for (std::size_t i = 0; i < n; ++i) {
        //    const double* q = data + i * dim;
//
        //    // ζητάμε k+1 για να πετάξουμε τον εαυτό του αν εμφανιστεί
        //    auto res = index.query_top(q, static_cast<std::size_t>(k) + 1);
//
        //    std::vector<std::int32_t> neigh;
        //    neigh.reserve(k);
        //    for (const auto& pr : res) {
        //        int id = pr.first;
        //        if (id == static_cast<int>(i)) continue; // πέτα τον εαυτό του
        //        neigh.push_back(id);
        //        if ((int)neigh.size() == k) break;
        //    }
//
        //    // αν δεν βρήκαμε αρκετούς, συμπληρώνουμε με -1
        //    while ((int)neigh.size() < k) {
        //        neigh.push_back(-1);
        //    }
//
        //    for (int j = 0; j < k; ++j) {
        //        all_ids[i * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)]
        //            = neigh[static_cast<std::size_t>(j)];
        //    }
//
        //    if ((i + 1) % 1000 == 0) {
        //        std::cerr << "\rComputed kNN for "
        //                  << (i + 1) << "/" << n << " points..." << std::flush;
        //    }
        //}
        //std::cerr << "\nFinished computing kNN." << std::endl;
        
        std::cerr << "Computing kNN for " << n_used << " points..." << std::endl;

        for (std::size_t i = 0; i < n_used; ++i) {
            const double* q = data + i * dim;

            auto res = index.query_top(q, static_cast<std::size_t>(k) + 1);

            std::vector<std::int32_t> neigh;
            neigh.reserve(k);
            for (const auto& pr : res) {
                int id = pr.first;
                if (id == static_cast<int>(i)) continue;              // πέτα τον εαυτό του
                if (id < 0 || (std::size_t)id >= n_used) continue;    // αγνόησε ids εκτός [0, n_used)
                neigh.push_back(id);
                if ((int)neigh.size() == k) break;
            }

            while ((int)neigh.size() < k) {
                neigh.push_back(-1);
            }

            for (int j = 0; j < k; ++j) {
                all_ids[i * static_cast<std::size_t>(k)
                        + static_cast<std::size_t>(j)]
                    = neigh[static_cast<std::size_t>(j)];
            }

            if ((i + 1) % 1000 == 0) {
                std::cerr << "\rComputed kNN for "
                          << (i + 1) << "/" << n_used << " points..." << std::flush;
            }
        }
        std::cerr << "\nFinished computing kNN." << std::endl;


        // 4. Γράψιμο binary: [int32 n][int32 k][n*k int32 ids]
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open output file " + out_path);
        }

        //std::int32_t n32 = static_cast<std::int32_t>(n);
        //std::int32_t k32 = static_cast<std::int32_t>(k);
//
        //out.write(reinterpret_cast<const char*>(&n32), sizeof(n32));
        //out.write(reinterpret_cast<const char*>(&k32), sizeof(k32));
        //out.write(
        //    reinterpret_cast<const char*>(all_ids.data()),
        //    all_ids.size() * sizeof(std::int32_t)
        //);

        std::int32_t n32 = static_cast<std::int32_t>(n_used);
        std::int32_t k32 = static_cast<std::int32_t>(k);

        out.write(reinterpret_cast<const char*>(&n32), sizeof(n32));
        out.write(reinterpret_cast<const char*>(&k32), sizeof(k32));
        out.write(
            reinterpret_cast<const char*>(all_ids.data()),
            all_ids.size() * sizeof(std::int32_t)
        );

        
        
        if (!out) {
            throw std::runtime_error("Error while writing output file " + out_path);
        }

        std::cerr << "Wrote kNN file: " << out_path << std::endl;
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 1;
    }
}
