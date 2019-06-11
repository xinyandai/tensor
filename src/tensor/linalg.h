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
Tensor<T, 2> _mm (
    const Tensor<T, 2> &a,
    const Tensor<T, 2> &b,
    Tensor<T, 2> *out,
    T alpha = 1.0f,
    T beta = 0.0f) {
//  alpha*op( A )*op( B ) + beta*C
//  code seems strange here since tensor is row based and BLAS is
//  column based. NT_A / NT_B is true if tensor a / b are already
//  transposed. We regard tensor contiguous if the tensor is just
//  transposed
  const bool NT_A = a.get_flag(FLAG_TRANSPOSED);
  const bool NT_B = b.get_flag(FLAG_TRANSPOSED);

  if (!a.get_flag(FLAG_CONTIGUOUS | FLAG_TRANSPOSED)) {
    return mm(a.as_contiguous(), b, out);
  }
  if (!b.get_flag(FLAG_CONTIGUOUS | FLAG_TRANSPOSED)) {
    return mm(a, b.as_contiguous(), out);
  }

//  M  specifies  the number  of rows  of the  matrix op( A )
//  and of the  matrix  C.  M  must  be at least  zero.
  FINTEGER M = a.shape()[0] ;
//  On entry,  N  specifies the number  of columns of the matrix
//  op( B ) and the number of columns of the matrix C.
  FINTEGER N = b.shape()[1];
//  On entry,  K  specifies  the number of columns of the matrix
//  op( A ) and the number of rows of the matrix op( B ).
  FINTEGER K = a.shape()[1];

  if (K != b.shape()[0] || M *N != out->size()) {
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
  const Tensor<T, 2> &b,
  Tensor<T, 2> *out,
  T alpha = 1.0f,
  T beta = 0.0f) {
  return _mm(b.transpose(), a.transpose(), out, alpha, beta);
}

template<typename T>
Tensor<T, 2> mm (
    const Tensor<T, 2> &a,
    const Tensor<T, 2> &b,
    T alpha = 1.0f,
    T beta = 0.0f) {

  Tensor<T, 2> out({a.shape()[0], b.shape()[1]});
  mm(a, b, &out, alpha, beta);
  return out;
}

template<typename T>
Tensor<size_type, 1> vq (
    const Tensor<T, 2> &a,
    const Tensor<T, 2> &b) {

  Tensor<T, 2> l2dist = l2_sqr(a, b);
  return arg_min(l2dist, 1);
}

template<typename T>
Tensor<T, 2> kmeans (const Tensor<T, 2> &x, 
  const size_type K, const size_type n_iter) {
  size_type N = x.shape()[0];
  size_type D = x.shape()[1];
  Tensor<T, 2> c = x[{S(0, K), S()}].as_contiguous();
  for (size_type iter = 0; iter < n_iter; iter++) {
    Tensor<size_type, 1> codes = vq(x, c);

    Tensor<T, 1> counter({K});
    counter.fill(0.f);
    c.fill(0.f);
    for (size_type i = 0; i < N; i++) {
      size_type code_idx = codes[{i}];
      c[{S(code_idx), S()}] += x[{S(i), S()}];
      counter[{code_idx}] += 1;
    }
    for (size_type i = 0; i < K; i++) {
      if (counter[{i}] > 0.0) {
        c[{S(i), S()}] /= counter[{i}];
      } else {
        std::cout << "[warning]: empty bucket at iteration "
          << iter + 1 << std::endl;
      }
    }
  }
  return c;
}

template<typename T>
Tensor<T, 1> mv (
    const Tensor<T, 2> &a, const T *b, Tensor<T, 1> * out) {

  size_type M = a.shape()[0], N = a.shape()[1];
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    out->data()[i] = fvec_inner_product(a.data()[i * N], b, N);
  }
}

template<typename T>
Tensor<T, 1> mv (const Tensor<T, 2> &a, const T *b) {
  Tensor<T, 1> out({a.shape()[0]});
  mm(a, *b, &out);
  return out;
}

template<typename T, size_type  D, class F >
Tensor<T, D-1> _reduce(
  const Tensor<T, D> &a, size_type axis, F f) {

  const Tensor<T, D> moved_a = a.move_axis(axis, D-1);
  std::array<size_type, D-1> shapes;
#pragma unroll
  for (size_type i = 0; i < D-1; ++i) {
    shapes[i] = moved_a.shape()[i];
  }
  Tensor<T, D-1> sqr(shapes);
  reduce_by_stride<T, D> (
      sqr.data(), moved_a.data(),
      sqr.stride().data(), moved_a.stride().data(),
      moved_a.shape().data(), f);
  return sqr;
}
template<typename T, size_type D, class F>
Tensor<size_type, D-1> _arg_reduce(
  const Tensor<T, D> &a, size_type axis, F f) {

  const Tensor<T, D> moved_a = a.move_axis(axis, D-1);
  std::array<size_type, D-1> shapes;
#pragma unroll
  for (size_type i = 0; i < D-1; ++i) {
    shapes[i] = moved_a.shape()[i];
  }
  Tensor<size_type, D-1> indices(shapes);
  arg_reduce_by_stride<T, D> (
      indices.data(), moved_a.data(),
      indices.stride().data(), moved_a.stride().data(),
      moved_a.shape().data(), f);
  return indices;
}

template<typename T, size_type  D>
Tensor<T, D-1> norm_sqr(
  const Tensor<T, D> &a, size_type axis=D-1) {
  return  _reduce(a, axis, norm_sqr_adder<T >());
}
template<typename T, size_type  D>
Tensor<T, D-1> max(
  const Tensor<T, D> &a, size_type axis=D-1) {
  return  _reduce(a, axis, max_assigner<T >());
}
template<typename T, size_type  D>
Tensor<T, D-1> min(
  const Tensor<T, D> &a, size_type axis=D-1) {
  return  _reduce(a, axis, min_assigner<T >());
}
template<typename T, size_type  D>
Tensor<size_type, D-1> arg_max(
  const Tensor<T, D> &a, size_type axis=D-1) {
  return _arg_reduce(a, axis, max_compare<T >());
}
template<typename T, size_type  D>
Tensor<size_type, D-1> arg_min(
  const Tensor<T, D> &a, size_type axis=D-1) {
  return _arg_reduce(a, axis, min_compare<T >());
}


template<typename T, size_type D>
Tensor<size_type, D> top_select(
  const Tensor<T, D> &a, size_type K,
  size_type axis=D-1, bool desc = false) {

  Tensor<T, D> moved_a = a.move_axis(axis, D-1);
  std::array<size_type, D> shapes;
#pragma unroll
  for (size_type i = 0; i < D-1; ++i) {
    shapes[i] = moved_a.shape()[i];
  }
  shapes[D-1] = K;
  Tensor<size_type , D> indices(shapes);
  if (K == moved_a.shape()[D-1]) {
    reorder_by_stride<T, D> (
      indices.data(), moved_a.data(),
      indices.stride().data(), moved_a.stride().data(),
      moved_a.shape().data(), K, desc, arg_sorter<T >());
  } else {
    reorder_by_stride<T, D> (
      indices.data(), moved_a.data(),
      indices.stride().data(), moved_a.stride().data(),
      moved_a.shape().data(), K, desc, top_selector<T >());
  }
  return indices;
}

template<typename T, size_type  D>
Tensor<T, D> arg_sort(
  const Tensor<T, D> &a, size_type axis=D-1, bool desc = false) {
  return top_select(a, a.shape()[axis], axis, desc);
}

template <typename T>
Tensor<T, 2> l2_sqr(
  const Tensor<T, 2>& a, const Tensor<T, 2>& b) {
  if (a.shape()[1] != b.shape()[1])
    throw std::runtime_error(
        "dimension do not match when calculating l2 sqr dist");
  const Tensor<T, 1> a_sqr = norm_sqr(a) ;
  const Tensor<T, 1> b_sqr = norm_sqr(b) ;
  Tensor<T, 2> m({a.shape()[0], b.shape()[0]});

#pragma omp parallel for
  for (int r = 0; r < m.shape()[0]; ++r) {
    for (int c = 0; c < m.shape()[1]; ++c) {
      m[{r, c}] = a_sqr[{r}] + b_sqr[{c}];
    }
  }

  tensor::mm(a, b.transpose(), &m, -2.0f, 1.0f);

  return m;
}



} // namespace tensor
#endif //TENSOR_CALCULATOR_H
