//
// Created by xinyan on 5/6/2019.
//
#pragma once
#ifndef TENSOR_STRDE_ITER_H
#define TENSOR_STRDE_ITER_H
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>

template <typename T>
class StrideIterator
{
  typedef int size_type;
 public:
  typedef T value_type;
  typedef T& reference;
  typedef size_type difference_type;
  typedef T* pointer;
  typedef std::random_access_iterator_tag iterator_category;

  StrideIterator(const StrideIterator& a) : ptr_(a.ptr_), step_(a.step_) { }
  StrideIterator(T* ptr, size_type step) : ptr_(ptr), step_(step) { }
  StrideIterator operator++() { ptr_+=step_; return *this; }
  StrideIterator operator--() { ptr_-=step_; return *this; }
  StrideIterator operator++(int) { StrideIterator i = *this; ptr_+=step_; return i; }
  StrideIterator operator--(int) { StrideIterator i = *this; ptr_-=step_; return i; }
  StrideIterator operator+=(size_type n) { ptr_ += n * step_; return *this; }
  StrideIterator operator-=(size_type n) { ptr_ -= n * step_; return *this; }

  T&  operator*() { return *ptr_; }
  T&  operator[](size_type n) { return ptr_[n*step_]; }
  T* operator->() { return ptr_; }
  bool operator==(const StrideIterator& rhs) { return ptr_ == rhs.ptr_; }
  bool operator!=(const StrideIterator& rhs) { return ptr_ != rhs.ptr_; }
  // friend operators
  friend bool operator<(const StrideIterator& x, const StrideIterator& y) {
    return x.ptr_ < y.ptr_;
  }
  friend size_type operator-(const StrideIterator& x, const StrideIterator& y) {
    return (x.ptr_ - y.ptr_) / x.step_;
  }
  friend StrideIterator operator+(const StrideIterator& x, size_type y) {
    StrideIterator c(x);
    c.ptr_ += y * c.step_;
    return c;
  }
  friend StrideIterator operator-(const StrideIterator& x, size_type y) {
    StrideIterator c(x);
    c.ptr_ -= y * c.step_;
    return c;
  }
  friend bool operator==(const StrideIterator& x, const StrideIterator& y) {
    return x.ptr_ == y.ptr_;
  }
  friend bool operator!=(const StrideIterator& x, const StrideIterator& y) {
    return x.ptr_ != y.ptr_;
  }
 private:
  T* ptr_;
  size_type step_;
};

#endif //TENSOR_STRDE_ITER_H
