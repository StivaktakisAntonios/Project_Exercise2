//L2 Function Euclidean Function
#include <cmath>
#include "L2.hpp"

//Vectors are and b, d is the dimension of them 
double l2_double_squared(const double* a, const double* b, size_t d){
  double sum = 0.0;
  for(size_t i = 0; i < d; i++){
    double dif; //Difference of a and b.
    dif = a[i]-b[i]; //for every dimension d we make the difference
    double square;
    square = dif*dif; // we keep the square of differnce in the sum variable
    sum += square;
  }
return sum; 
}


double l2_float_squared(const float* a, const float* b, size_t d){
  double sum = 0.0;
  for(size_t i = 0; i < d; i++){
    double dif;
    dif = a[i] - b[i];
    double square;
    square = dif*dif;
    sum += square;
  }
  return sum;
}


//Here we calculate the sqrt of square which we claculate on upper functions.
double l2_double(const double* a, const double* b, size_t d){
  double sum;
  sum = l2_double_squared(a, b, d);
  double root;
  root = sqrt(sum);
  return root;
}

double l2_float(const float* a, const float* b, size_t d){
  double sum;
  sum = l2_float_squared(a, b, d);
  double root;
  root = sqrt(sum);
  return root;
}