//
// Created by xinyan on 11/5/2019.
//
#pragma once
#ifndef TENSOR_UTIL_H
#define TENSOR_UTIL_H
#include <array>
#define MAX_ELEMENT (1<<30) // 2147483647
#define SLICE_END (MAX_ELEMENT) // 2147483647
namespace tensor {

typedef int size_type;


template<typename size_type, size_type D>
size_type MULTIPLIER(const std::array <size_type, D> &a) {
  size_type result = 1;
#pragma unroll
  for (int i = 0; i < D; ++i) {
    result *= a[i];
  }
  return result;
}

template<typename T, size_type N>
static void
dump(const T *data, const size_type *stride,
     const size_type *shape) {
  if constexpr (N == 1) {
    for (int i = 0; i < shape[0]; ++i) {
      std::cout << *data << "\t ";
      data += *stride;
    }
    std::cout << std::endl;
  } else if constexpr (N > 1) {
    for (int i = 0; i < shape[0]; ++i) {
      dump<T, N - 1>(data, stride + 1, shape + 1);
      data += *stride;
    }
  }
}

template<typename T>
struct adder {
  void operator()(T &a, const T &b) { a += b; }
};
template<typename T>
struct subtract {
  void operator()(T &a, const T &b) { a -= b; }
};
template<typename T>
struct divider {
  void operator()(T &a, const T &b) { a /= b; }
};
template<typename T>
struct assignment {
  void operator()(T &a, const T &b) { a = b; }
};

template<typename T>
struct multiplier {
  void operator()(T &a, const T &b) { a *= b; }
};
template<typename T, size_type data_N, size_type source_N = 0, typename F>
static void
operation_by_stride(T *data, const T *source,
                    const size_type *stride_data,
                    const size_type *stride_source,
                    const size_type *shape,
                    F f) {
  static_assert(data_N >= source_N);
  if constexpr (data_N == 1) {
    for (int i = 0; i < *shape; ++i) {

      f(*data, *source);

      data += *stride_data;
      if constexpr (source_N > 0) {
        source += *stride_source;
      }
    }
  } else if constexpr (data_N > 1) {
    for (int i = 0; i < *shape; ++i) {

      if constexpr (source_N > 0) {
        operation_by_stride<T, data_N - 1, source_N - 1>(
            data, source, stride_data + 1, stride_source + 1, shape + 1, f);
        data += *stride_data;
        source += *stride_source;
      } else {
        operation_by_stride<T, data_N - 1, 0>(
            data, source, stride_data + 1, stride_source + 1, shape + 1, f);
        data += *stride_data;
      }
    }
  }
}

struct Slice {
  size_type begin_;
  size_type end_;
  size_type step_;
  void set_shape(size_type shape) {
    if (step_ == 0) {
      throw std::runtime_error("slice step size is 0");
    }
    int end = end_ >= 0? end_ : shape + end_;
    if (end_ < 0)
      end_ += shape;
    else if (end_ == SLICE_END)
      end_ = shape;
    if (begin_ < 0)
      begin_ += shape;
  }
  size_type slice_size() {
    if (end_ - begin_ > 0 && step_ > 0) {
      return (end_ - begin_) / step_;
    }
    if (end_ - begin_ < 0 && step_ < 0) {
      return (begin_ - end_) / (- step_);
    }
    char message[1024];
    sprintf(message, "slice step size %d is not compatible "
                     "with slice interval [%d, %d]",
                     step_, begin_, end_);
    throw std::runtime_error(message);
  }
  size_type slice_offset(size_type stride) {
    return begin_ * stride;
  }
  size_type stride_size(size_type stride) {
    return stride * step_;
  }
  explicit Slice(): begin_(0), end_(SLICE_END), step_(1) {};
  explicit Slice(size_type begin)
      :begin_(begin), end_(SLICE_END), step_(1) {};
  explicit Slice(size_type begin, size_type end)
    :begin_(begin), end_(end), step_(1) {};
  explicit Slice(size_type begin, size_type end, size_type step)
    :begin_(begin), end_(end), step_(step) {};
};

} // namespace tensor
#endif //TENSOR_UTIL_H
