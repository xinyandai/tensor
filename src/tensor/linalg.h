//
// Created by xinyan on 30/4/2019.
//
#pragma once
#ifndef TENSOR_CALCULATOR_H
#define TENSOR_CALCULATOR_H
#include "tensor.h"
#include "simd.h"

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
//  code seems strange here since blas is row based and tensor is
//  column based. NT_A / NT_B is true if tensor a / b are already
//  transposed. We regard tensor contiguous if the tensor is just
//  transposed
  const bool NT_A = a.get_flag(FLAG_TRANSPOSED);
  const bool NT_B = b.get_flag(FLAG_TRANSPOSED);

  if ((!a.get_flag(FLAG_CONTIGUOUS)) && (!NT_A)) {
    return mm(a.as_contiguous(), b, out);
  }
  if ((!b.get_flag(FLAG_CONTIGUOUS)) && (!NT_B)) {
    return mm(a, b.as_contiguous(), out);
  }
  //  alpha*op( A )*op( B ) + beta*C
  T alpha = 1.0f;
  T beta = 0.0f;
//  M  specifies  the number  of rows  of the  matrix op( A )
//  and of the  matrix  C.  M  must  be at least  zero.
  FINTEGER M = a.shape()[0] ;
//  On entry,  N  specifies the number  of columns of the matrix
//  op( B ) and the number of columns of the matrix C.
  FINTEGER N = b.shape()[1];
//  On entry,  K  specifies  the number of columns of the matrix
//  op( A ) and the number of rows of the matrix op( B ).
  FINTEGER K = a.shape()[1];

  if (K != b.shape()[0] || M != out->shape()[0]
  || N != out->shape()[1]) {
    throw std::runtime_error(
        "shape not matched in matrix multiplication.");
  }
//  On entry, LDA specifies the first dimension of A as declared
//  in the calling (sub) program. When  TRANSA = 'N' or 'n' then
//  LDA must be at least  max( 1, m ), otherwise  LDA must be at
//  least  max( 1, k ).
  FINTEGER LDA = NT_A ?  M : K;
//  On entry, LDB specifies the first dimension of B as declared
//  in the calling (sub) program. When  TRANSB = 'N' or 'n' then
//  LDB must be at least  max( 1, k ), otherwise  LDB must be at
//  least  max( 1, n ).
  FINTEGER LDB = NT_B ?  K : N;
//  On entry, LDC specifies the first dimension of C as declared
//  in  the  calling  (sub)  program.   LDC  must  be  at  least
//  max( 1, m ).
  FINTEGER LDC = M;

  if constexpr (std::is_same<T, float >::value) {
    sgemm_(NT_A ? "N" : "T", NT_B ? "N" : "T", &M, &N, &K,
        &alpha, a.data(), &LDA, b.data(), &LDB,
        &beta, out->data(), &LDC);
  } else if constexpr (std::is_same<T, double >::value) {
    dgemm_(NT_A ? "N" : "T", NT_B ? "N" : "T", &M, &N, &K,
           &alpha, a.data(), &LDA, b.data(), &LDB,
           &beta, out->data(), &LDC);
  }

  return *out;
}

template<typename T>
Tensor<T, 2> mm (
    const Tensor<T, 2> &a,
    const Tensor<T, 2> &b) {

  Tensor<T, 2> out({a.shape()[0], b.shape()[1]});
  mm(a, b, &out);
  return out;
}

} // namespace tensor
#endif //TENSOR_CALCULATOR_H
