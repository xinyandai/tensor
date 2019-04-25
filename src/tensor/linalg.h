//
// Created by xinyan on 30/4/2019.
//
#pragma once
#ifndef TENSOR_CALCULATOR_H
#define TENSOR_CALCULATOR_H
#include "tensor.h"

#ifndef FINTEGER
#define FINTEGER long
#endif
extern "C" {
/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */
int
sgemm_ (
    const char *transa, const char *transb,
    FINTEGER *m, FINTEGER * n, FINTEGER *k,
    const float *alpha, const float *a,
    FINTEGER *lda, const float *b, FINTEGER *ldb,
    float *beta, float *c, FINTEGER *ldc);

int dgemm_(
    char *transa, char *transb,
    FINTEGER *m, FINTEGER *n, FINTEGER *k,
    const double *alpha, const double *a,
    FINTEGER *lda, const double *b, FINTEGER *ldb,
    const double *beta, const double *c, FINTEGER *ldc);
}


namespace tensor {
template<typename T>
Tensor<T, 2> mm (
    const Tensor<T, 2> &a,
    const Tensor<T, 2> &b) {
  Tensor<T, 2> contiguous_a = a.as_contiguous();
  Tensor<T, 2> contiguous_b = b.as_contiguous();
  contiguous_a.dump();
  contiguous_b.dump();
//  M  specifies  the number  of rows  of the  matrix op( A )
//  and of the  matrix  C.  M  must  be at least  zero.
  FINTEGER M = a.shape()[0];
//  On entry,  N  specifies the number  of columns of the matrix
//  op( B ) and the number of columns of the matrix C.
  FINTEGER N = b.shape()[1];
//  On entry,  K  specifies  the number of columns of the matrix
//  op( A ) and the number of rows of the matrix op( B ).
  FINTEGER K = a.shape()[1];

//  alpha*op( A )*op( B ) + beta*C
  Tensor<T, 2> out({M, N});
  T alpha = 1.0f;
  T beta = 0.0f;
  std::cout << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  if constexpr (std::is_same<T, float >::value) {
    sgemm_("N", "N", &M, &N, &K,
           &alpha, a.data(),
           &M, b.data(), &K,
           &beta, out.data(), &M);
  } else if constexpr (std::is_same<T, double >::value) {
    dgemm_("N", "N", &M, &N, &K,
           &alpha, a.data(),
           &M, b.data(), &K,
           &beta, out.data(), &M);
  }
  std::cout << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  return out;
}
} // namespace tensor
#endif //TENSOR_CALCULATOR_H
