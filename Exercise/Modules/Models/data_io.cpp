#include "data_io.hpp"
#include <fstream>
#include <stdexcept>
#include <cmath>

namespace {
    // Convera big-endian -> host for 32-bit (MNIST headers)
    static uint32_t be32_to_host(uint32_t x){
        unsigned char* p = reinterpret_cast<unsigned char*>(&x);
        return (uint32_t(p[0])<<24) | (uint32_t(p[1])<<16) | (uint32_t(p[2])<<8) | uint32_t(p[3]);
    }
}

// MNIST (IDX big-endian) images only 
void load_mnist_images(const std::string& path, std::vector<double>& out, std::size_t& n, std::size_t& dim)
{
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+path);

    uint32_t magic_be=0, num_be=0, rows_be=0, cols_be=0;
    f.read(reinterpret_cast<char*>(&magic_be),4);
    f.read(reinterpret_cast<char*>(&num_be),4);
    f.read(reinterpret_cast<char*>(&rows_be),4);
    f.read(reinterpret_cast<char*>(&cols_be),4);

    const uint32_t magic = be32_to_host(magic_be);
    const uint32_t num = be32_to_host(num_be);
    const uint32_t rows = be32_to_host(rows_be);
    const uint32_t cols = be32_to_host(cols_be);

    if (magic != 2051) 
        throw std::runtime_error("MNIST magic != 2051 for "+path);

    n = static_cast<std::size_t>(num);
    dim = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    out.resize(n*dim);

    std::vector<unsigned char> buf(dim);
    for(std::size_t i=0;i<n;i++){
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(dim));
        if(!f) 
            throw std::runtime_error("short read in "+path);
        for(std::size_t j=0;j<dim;j++){
            out[i*dim+j] = static_cast<double>(buf[j]); /// 255.0; // normalize to [0,1]
        }
    }
}

// SIFT (fvecs little-endian)
void load_sift_fvecs(const std::string& path, std::vector<double>& out, std::size_t& n, std::size_t& dim)
{
    std::ifstream f(path, std::ios::binary);
    if(!f) 
        throw std::runtime_error("cannot open "+path);

    // get file size
    f.seekg(0, std::ios::end);
    const auto sz = f.tellg();
    f.seekg(0, std::ios::beg);

    // read first dim
    int32_t d_first=0;
    f.read(reinterpret_cast<char*>(&d_first),4);
    if(!f) 
        throw std::runtime_error("short read in "+path);
    if (d_first <= 0) 
        throw std::runtime_error("invalid first dim in fvecs: "+path);

    dim = static_cast<std::size_t>(d_first);
    const std::size_t stride = 4 + dim*4; // (int32 dim) + dim * float32

    // check file size consistency and compute n
    if (sz % static_cast<std::streamoff>(stride) != 0) {
        throw std::runtime_error("fvecs file size is not a multiple of record stride: " + path);
    }
    n = static_cast<std::size_t>(sz) / stride;

    out.resize(n*dim);

    // rewind and read all vectors
    f.clear();
    f.seekg(0, std::ios::beg);
    for(std::size_t i=0;i<n;i++){
        int32_t d=0;
        f.read(reinterpret_cast<char*>(&d),4);
        if(!f) 
            throw std::runtime_error("short read in "+path);
        if (d != static_cast<int32_t>(dim))
            throw std::runtime_error("mixed dims in fvecs: "+path);

        for(std::size_t j=0;j<dim;j++){
            float v;
            f.read(reinterpret_cast<char*>(&v),4);
            if(!f) 
                throw std::runtime_error("short read in "+path);
            out[i*dim+j] = static_cast<double>(v);
        }
    }
    // --- NEW Optional normalization step for SIFT vectors ---
    for (std::size_t i = 0; i < n; ++i) {
        double norm = 0.0;
        for (std::size_t j = 0; j < dim; ++j)
            norm += out[i * dim + j] * out[i * dim + j];
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (std::size_t j = 0; j < dim; ++j)
                out[i * dim + j] /= norm;
        }
    }
}