//
// Created by xinyan on 25/4/2019.
//
#include <iostream>
#include "../tensor/helper.h"
#include "../tensor/tensor.h"
#include "../tensor/linalg.h"

using namespace tensor;

int main() {
  std::cout << "add example" << std::endl;
  tensor::Tensor<float, 2> a({3, 4});
  tensor::Tensor<float, 2> b({4, 5});
  a.fill(0.5);
  b.fill(0.4);

  tensor::Tensor<float, 2> c = tensor::mm(a, b);
  c.dump();
}
