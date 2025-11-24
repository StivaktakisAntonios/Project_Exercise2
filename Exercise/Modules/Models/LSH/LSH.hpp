//We are going to state the LSH interface here.
#pragma once
#include <vector>
#include <cstddef> //gia size_t
#include <random> //For mt19937
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

class L2Hash{
  private:
    std::size_t dim; //Dimensional for check tou dianismatos px dim = 2 -- A(x,y)
    double w; //width/length bucket w>0 -- platos kathe kouva, oso megalitero toso pio polloi yposifioi tha pesoyn mesa
    double inv_w; //save 1.0/w for faster applying 
    double b; //metatopisi apo U[0,w] omoiomorfi metatopisi gia thn hash 
    std::vector<double> a; //vector of real provoli me N(0,1) ana sinistosa provoles gia thn hash poy tha pesoyn ston koyva
  
  

  public:
    //Random Constructor
    L2Hash(std::size_t dim, double w, std::mt19937& rng);

    //Deterministic constructor
    L2Hash(const std::vector<double>& a, double b, double w);

    //Efarmogi hash akeraiou kadou
    long long hash_function (const double* x, std::size_t d) const; //x has d length, apply makes dot in double and return long long

};


//klasi pou pairnei oles tis hash functions kai epistrefi ena kleidi gia ton pinaka lsh
class Signature {
  private:
  std::vector<L2Hash> hashes; //pinakas me hashes functions
  std::size_t dim; //diastasi
  std::size_t k; //plithos hashes
  double w; //platos kouva

  public:
  //constructor random
  Signature(std::size_t k, std::size_t dim, double w, std::mt19937& rng);

  //signature_function epistrefei ena bucket ana hash 
  std::vector<long long> signature_function(const double* x, std::size_t d) const;

  //enoni tous k akeraious me |
  std::string make_key(const std::vector<long long>& sig)const;


  std::string key_for(const double* x, std::size_t d) const;
};



//klasi LSH Table krataei signature pinakes dld antistixizoume ta signature px 0|0|1 -> me simia px A, B ..
class LSHTable {
  private:
    Signature g; //to g function tou pinaka
    std::unordered_map<std::string, std::vector<int>> buckets; //edo einai to map poy tha einai oi antistixies me ta strings keys kai ta vectors 
    //kleidi = to string apo signature::make key
    //timi = lista ap;o IDs simion pou epesan se ayton ton kado

    std::size_t dim; // gia elegxo oti ta eiserxomean dianismata exoun sosti diastasi

  public:
  LSHTable() = delete;
  //constructor
  LSHTable(std::size_t k, std::size_t dim, double w, std::mt19937& rng);


  void add_point(int id, const double* x, std::size_t d);


  std::vector<int> candidates(const double* q, std::size_t d) const;
};




class LSHIndex{
  private:
    std::vector<LSHTable> lsh_tables;
    std::size_t dim;
    std::size_t L;
    double w;
    std::size_t k; //arithmos hashes poses dld L2hash

    const double* data_ptr = nullptr;
    std::size_t n_points = 0;
    
  public:
  //constructor
  LSHIndex(std::size_t dim, std::size_t k, std::size_t L, double w, std::mt19937& rng);

  void build(const double* data, std::size_t n); //data exei mikos n * dim se row major 
  
  std::vector<std::pair<int,double>> query_top(const double* q, std::size_t d, std::size_t N);

  std::vector<int> query_range(const double* q, std::size_t d, double R);
  
};