//
// Created by xinyan on 24/4/2019.
//
#pragma once
#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#define CHECK_SHAPE (true)
#define FLAG_CONTIGUOUS (1<<0)

#include <array>
#include <algorithm>
#include <memory>
#include <type_traits>
#include "helper.h"

namespace tensor {

template<typename T, size_type D, bool HOST=true>
class Tensor {
 private:
    template <size_type N, size_type R=D>
    void _continuous_stride() {
      static_assert(N <= R);
      static_assert(N >= 0);
      if constexpr (N > 0 ) {
        if constexpr (N == R ) {
          stride_[N-1] = 1;
        } else {
          stride_[N-1] = stride_[N] * shape_[N];
        }
        return _continuous_stride< N - 1, R>();
      }
    }

  template <size_type R>
  void _check_shape (const Tensor<T, R, HOST> &t) {
    static_assert(R <= D);
    if constexpr CHECK_SHAPE {
      for (int i = 0; i < R; ++i) {
        if(shape_[i]!=t.shape()[i]) {
          char message[1024];
          sprintf(message, "Not doable for different "
                           "shape at dim %d from %d to %d",
                           i, shape_[i], t.shape()[i]);
          throw std::runtime_error(message);
        }
      }
    }
  }

 public:
  Tensor<T, D, HOST> operator[] (std::array<Slice, D> slices) {
    Tensor<T, D, HOST> result;
    result.shape_ = {0};
    result.data_= data_;
#pragma unroll
    for (int i = 0; i < D; ++i) {
      slices[i].set_shape(shape_[i]);
    }
#pragma unroll
    for (int i = 0; i < D; ++i) {
      result.shape_[i] = slices[i].slice_size();
    }
#pragma unroll
    for (int i = 0; i < D; ++i) {
      result.data_ += slices[i].slice_offset(stride_[i]);
    }
#pragma unroll
    for (int i = 0; i < D; ++i) {
      result.stride_[i] = slices[i].stride_size(stride_[i]);
    }
    result.size_ = MULTIPLIER<size_type, D>(result.shape_);
    result.ptr_= ptr_;
    return result;
  }
  T operator[] (std::array<size_type , D> slices) {
    T *offset = data_;
#pragma unroll
    for (int i = 0; i < D; ++i) {
      offset += (stride_[i] * slices[i]);
    }
    return *offset;
  }

  explicit Tensor()
      :
      size_(0),
      shape_({0}),
      data_(nullptr),
      ptr_(data_),
      flag_(0)  {}

  explicit Tensor(
      const std::array<size_type, D> & shapes)
      :
      size_(MULTIPLIER<size_type, D>(shapes)),
      shape_(shapes),
      data_(new T[size_]),
      ptr_(data_),
      flag_(FLAG_CONTIGUOUS) {

    this->_continuous_stride<D>();
  }

  // reshape
  template <size_type R>
  Tensor(Tensor<T, R, HOST> &t,
      const std::array<size_type, D> & shapes)
      :
      size_(MULTIPLIER<size_type, D>(shapes)),
      shape_(shapes),
      data_(t.data()),
      ptr_(t.ptr()),
      flag_(FLAG_CONTIGUOUS & (D == R))  {

    static_assert(D >= R);
    if (size_ == t.size()) {
      this->_continuous_stride<D>();
    } else if (size_ > t.size() && size_ % t.size() == 0) {
      this->_continuous_stride<R, R>();
#pragma unroll
      for (int i = R; i < D; ++i) {
        stride_[i] = 0;
      }
    } else {
      throw std::runtime_error(
          "construction failed for un-compatible shape.");
    }
  }

  // copying constructor
  Tensor(const Tensor<T, D, HOST> &t):
    size_(t.size_),
    stride_(t.stride_),
    shape_(t.shape_),
    data_(t.data_),
    ptr_(t.ptr_) ,
    flag_(t.flag_) {}

  Tensor(const Tensor<T, D, HOST> &t,
      size_type axis_1, size_type axis_2) :
      size_(t.size_),
      stride_(t.stride_),
      shape_(t.shape_),
      data_(t.data_),
      ptr_(t.ptr_),
      flag_(t.flag_ & (!FLAG_CONTIGUOUS) )  {
    shape_[axis_1] = t.shape_[axis_2];
    shape_[axis_2] = t.shape_[axis_1];
    stride_[axis_1] = t.stride_[axis_2];
    stride_[axis_2] = t.stride_[axis_1];
  }

  // Move constructor.
  Tensor(const Tensor<T, D, HOST>&& t) noexcept
  : size_(t.size_),
    stride_(t.stride_),
    shape_(t.shape_),
    data_(t.data_),
    ptr_(std::move(t.ptr_)) ,
    flag_(t.flag_) {}

  // Move assignment operator.
  Tensor& operator = (Tensor<T, D, HOST>&& t) noexcept {
    size_ = t.size_;
    stride_ = t.stride_;
    shape_ = t.shape_;
    data_ = t.data_;
    ptr_ = std::move(t.ptr_);
    flag_ = t.flag_;
    return *this;
  }

  // assignment operator
  Tensor& operator = (T t) {
    fill(t);
    return *this;
  }
  Tensor& operator = (const Tensor<T, D, HOST> &t) {
    _check_shape(t);
    operation_by_stride<T, D, D >(
        data_, t.data_, stride_.data(), t.stride_.data(),
        shape_.data(), assignment<T >());
    return *this;
  }
  template <size_type R>
  Tensor& operator = (const Tensor<T, R, HOST> &t) {
    _check_shape(t);
    operation_by_stride<T, D, R >(
        data_, t.data(), stride_.data(), t.stride().data(),
        shape_.data(), assignment<T >());
  }
  template <size_type R>
  Tensor& operator += (const Tensor<T, R, HOST> &t) {
    _check_shape(t);
    operation_by_stride<T, D, R >(
        data_, t.data(), stride_.data(), t.stride().data(),
        shape_.data(), adder<T >());
    return *this;
  }
  template <size_type R>
  Tensor& operator -= (const Tensor<T, R, HOST> &t) {
    _check_shape(t);
    operation_by_stride<T, D, R >(
        data_, t.data(), stride_.data(), t.stride().data(),
        shape_.data(), subtract<T >());
    return *this;
  }
  template <size_type R>
  Tensor& operator *= (const Tensor<T, R, HOST> &t) {
    _check_shape(t);
    operation_by_stride<T, D, R >(
        data_, t.data(), stride_.data(), t.stride().data(),
        shape_.data(), multiplier<T >());
    return *this;
  }
  template <size_type R>
  Tensor& operator /= (const Tensor<T, R, HOST> &t) {
    _check_shape(t);
    operation_by_stride<T, D, R >(
        data_, t.data(), stride_.data(), t.stride().data(),
        shape_.data(), divider<T >());
    return *this;
  }
  template <size_type R>
  Tensor<T, D, HOST> operator+ (const Tensor<T, R, HOST> &t) {
    Tensor<T, D, HOST> out(shape_);
    out = *this;
    out += t;
    return out;
  }
  template <size_type R>
  Tensor<T, D, HOST> operator- (const Tensor<T, R, HOST> &t) {
    Tensor<T, D, HOST> out(shape_);
    out = *this;
    out -= t;
    return out;
  }
  template <size_type R>
  Tensor<T, D, HOST> operator* (const Tensor<T, R, HOST> &t) {
    Tensor<T, D, HOST> out(shape_);
    out = *this;
    out *= t;
    return out;
  }
  template <size_type R>
  Tensor<T, D, HOST> operator/ (const Tensor<T, R, HOST> &t) {
    Tensor<T, D, HOST> out(shape_);
    out = *this;
    out /= t;
    return out;
  }
  ~ Tensor() = default;

  template <size_type R>
  inline Tensor<T, R, HOST>
  broadcast_to (const std::array<size_type, R> & shapes) {
    return (Tensor<T, R, HOST>(*this, shapes));
  }
  template <size_type R>
  inline Tensor<T, R, HOST>
  reshape (const std::array<size_type, R> & shapes) {
    return (Tensor<T, R, HOST>(*this, shapes));
  }

  Tensor<T, D, HOST>
  swap_axis (size_type axis_1, size_type axis_2) {
    Tensor<T, D, HOST> copied(*this, axis_1, axis_2);
    return (copied);
  }

  Tensor<T, D, HOST>
  transpose () {
    return this->swap_axis(0, D-1);
  }

  void fill(T value) {
    if (get_flag(FLAG_CONTIGUOUS)) {
      for (int i = 0; i < size_; ++i)
        data_[i] = value;
    }
    else {
      operation_by_stride<T, D, 0>(
          data_, &value, stride_.data(), NULL,
          shape_.data(), assignment<T >());
    }
  }

  void dump() const {
    tensor::dump<T, D>(data_, stride_.data(), shape_.data());
    std::cout << std::endl;
  }

  Tensor<T, D, HOST> as_contiguous() const {
    if (get_flag(FLAG_CONTIGUOUS)) {
      return *this;
    } else {
      Tensor<T, D, HOST> copied(this->shape());
      T *data = copied.data();
      operation_by_stride<T, D, D>(
          copied.data(), data_, copied.stride_.data(), stride_.data(),
          shape_.data(), assignment<T >());
      return copied;
    }
  }

  const std::shared_ptr<T[]> &ptr() const {return ptr_;}

  const T* data() const {return data_;}
  T* data() {return data_;}

  bool get_flag(int flag) const {return (flag_ & flag) > 0; }
  void set_flag(int flag) { flag_ |= flag; }
  void unset_flag(int flag) { flag_ &= (!flag); }

  static constexpr size_type dim() {return D;}
  size_type size() const {return size_;}

  const std::array<size_type, D > &
      shape() const {return shape_;}
  const std::array<size_type, D > &
      stride() const {return stride_;}

 protected:
  size_type size_;
  // Tuple to step in each dimension when traversing an array.
  std::array<size_type, D> stride_;
  std::array<size_type, D> shape_;
  T *data_;
  std::shared_ptr<T[]> ptr_;
  int flag_;
};


template<typename T, size_type D,  bool HOST>
Tensor<size_type, D, HOST> argsort (
    const Tensor<T, D, HOST> &a,
    size_type axis,
    Tensor<size_type, D, HOST> *out = NULL) {}

template<typename T, size_type D,  bool HOST>
Tensor<size_type, D - 1, HOST> argmin (
    const Tensor<T, D, HOST> &a,
    size_type axis,
    Tensor<size_type, D - 1, HOST> *out = NULL) {}

template<typename T, size_type D,  bool HOST>
Tensor<T, D - 1, HOST> min (
    const Tensor<T, D, HOST> &a,
    size_type axis,
    Tensor<T, D - 1, HOST> *out = NULL) {}

// tensor calculation
template<typename T, size_type D,  bool HOST, class B>
Tensor<T, D, HOST> add (
    const Tensor<T, D, HOST> &a,
    const B &b,
    Tensor<T, D, HOST> *out) {

  (*out) = a;
  (*out) += b;
  return *out;
}

template<typename T, size_type D,  bool HOST, class B>
Tensor<T, D, HOST> multiply (
    const Tensor<T, D, HOST> &a,
    const B &b,
    Tensor<T, D, HOST> *out) {

  (*out) = a;
  (*out) *= b;
  return *out;
}

template<typename T, size_type D,  bool HOST, class B>
Tensor<T, D, HOST> divide (
    const Tensor<T, D, HOST> &a,
    const B &b,
    Tensor<T, D, HOST> *out) {

  (*out) = a;
  (*out) /= b;
  return *out;
}

} ; // namespace tensor
#endif //TENSOR_TENSOR_H
