//We will implement 4 functions 2 for double and 2 for float numbers.
//We have two types of L2 functions:
//One without root which is faster
//The other one with sqrt root which is for better results.

#include <cstddef> //for size_t


//Vector a vector b and the dimension of vectors.
//Squared is without sqrt and is faster
double l2_double_squared(const double* a, const double* b, size_t d);
double l2_float_squared(const float* a, const float* b, size_t d);


//Here we calculate also the sqrt of square and it is better for results.
double l2_double(const double* a, const double* b, size_t d);
double l2_float(const float* a, const float* b, size_t d);