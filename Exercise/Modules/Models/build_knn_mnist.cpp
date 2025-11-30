#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

#include "Template/data_io.hpp"
#include "IVFFlat/ivfflat.hpp"

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " -d <input.idx3-ubyte> -k <knn> -o <output.bin>\n";
}

int main(int argc, char** argv) {
    std::string data_path;
    std::string out_path;
    int k = 10;

    // Parse arguments
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
        // 1. Load MNIST dataset
        std::vector<double> data_vec;
        std::size_t n = 0, dim = 0;
        load_mnist_images(data_path, data_vec, n, dim);
        const double* data = data_vec.data();

        std::cerr << "Loaded MNIST dataset: n=" << n
                  << ", dim=" << dim << std::endl;

        // Use all points for MNIST (typically 60k)
        std::size_t n_used = n;
        std::cerr << "Using all " << n_used
                  << " points to build k-NN graph." << std::endl;

        if ((std::size_t)k >= n_used) {
            k = static_cast<int>(n_used) - 1;
        }

        // 2. IVFFlat parameters (adjust as needed)
        IVFFlat::Params P;
        P.dim       = dim;
        P.kclusters = 256;      // fewer clusters for MNIST (60k points)
        P.nprobe    = 10;       // how many clusters to probe at query time
        P.max_iters = 10;       // k-means iterations
        P.tol       = 1e-3;     // tolerance
        P.seed      = 1;        // seed

        IVFFlat index(P);
        std::cerr << "Building IVFFlat index..." << std::endl;
        index.build(data, n);
        std::cerr << "Index built." << std::endl;

        // 3. Compute approximate k-NN for each point
        std::vector<std::int32_t> all_ids;
        all_ids.resize(n_used * static_cast<std::size_t>(k));

        std::cerr << "Computing kNN for " << n_used << " points..." << std::endl;

        for (std::size_t i = 0; i < n_used; ++i) {
            const double* q = data + i * dim;

            auto res = index.query_top(q, static_cast<std::size_t>(k) + 1);

            std::vector<std::int32_t> neigh;
            neigh.reserve(k);
            for (const auto& pr : res) {
                int id = pr.first;
                if (id == static_cast<int>(i)) continue;              // skip self
                if (id < 0 || (std::size_t)id >= n_used) continue;    // skip invalid ids
                neigh.push_back(id);
                if ((int)neigh.size() == k) break;
            }

            // Fill with -1 if not enough neighbors found
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

        // 4. Write binary file: [int32 n][int32 k][n*k int32 ids]
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open output file " + out_path);
        }

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
