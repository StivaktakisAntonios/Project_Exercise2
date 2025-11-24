#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>



//load  MNIST images (IDX big-endian).fill row-major 

void load_mnist_images(const std::string& path, std::vector<double>& out, std::size_t& n, std::size_t& dim);



//load SIFT fvecs little endian float 32

void load_sift_fvecs(const std::string& path, std::vector<double>& out, std::size_t& n, std::size_t& dim);