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
    const Tensor<T, 2> &b,
    Tensor<T, 2> *out) {
  if ((!a.get_flag(FLAG_CONTIGUOUS)) && (!a.get_flag(FLAG_TRANSPOSED))) {
    return mm(a.as_contiguous(), b, out);
  }
  if ((!b.get_flag(FLAG_CONTIGUOUS)) && (!b.get_flag(FLAG_TRANSPOSED))) {
    return mm(a, b.as_contiguous(), out);
  }
  const char* T_A = a.get_flag(FLAG_CONTIGUOUS) ? "N" : "T";
  const char* T_B = a.get_flag(FLAG_CONTIGUOUS) ? "N" : "T";
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
  T alpha = 1.0f;
  T beta = 0.0f;

  if constexpr (std::is_same<T, float >::value) {
    sgemm_(T_A, T_B, &M, &N, &K,
           &alpha, a.data(),
           &M, b.data(), &K,
           &beta, out->data(), &M);
  } else if constexpr (std::is_same<T, double >::value) {
    dgemm_(T_A, T_B, &M, &N, &K,
           &alpha, a.data(),
           &M, b.data(), &K,
           &beta, out->data(), &M);
  }

  return *out;
}

template<typename T>
Tensor<T, 2> mm (
    const Tensor<T, 2> &a,
    const Tensor<T, 2> &b) {
//  M  specifies  the number  of rows  of the  matrix op( A )
//  and of the  matrix  C.  M  must  be at least  zero.
  FINTEGER M = a.shape()[0];
//  On entry,  N  specifies the number  of columns of the matrix
//  op( B ) and the number of columns of the matrix C.
  FINTEGER N = b.shape()[1];
//  alpha*op( A )*op( B ) + beta*C
  Tensor<T, 2> out({M, N});
  mm(a, b, &out);
  return out;
}

} // namespace tensor
#endif //TENSOR_CALCULATOR_H
