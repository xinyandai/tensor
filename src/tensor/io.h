//
// Created by xinyan on 12/5/2019.
//
#pragma once
#ifndef TENSOR_IO_H
#define TENSOR_IO_H

#include <iostream>
#include <fstream>
#include "tensor.h"

namespace tensor {

template <typename T>
Tensor<T, 2, true>
fvecs(const char* fvecs) {
    std::ifstream fin(fvecs,
        std::ios::binary | std::ios::ate);
    if (!fin) {
        char message[1024];
        sprintf(message, "cannot open file %s", fvecs);
        throw std::runtime_error(message);
    }

    size_t fileSize = fin.tellg();
    fin.seekg(0, fin.beg);
    if (fileSize == 0) {
        char message[1024];
        sprintf(message, "File size is 0 %s", fvecs);
        throw std::runtime_error(message);
    }

    int dim;
    fin.read(reinterpret_cast<char*>(&dim), sizeof(int));

    size_t bytesPerRecord = dim * sizeof(T) + 4;
    if (fileSize % bytesPerRecord != 0) {
        char message[1024];
        sprintf(message, "File not aligned  [%s]", fvecs);
        throw std::runtime_error(message);
    }

    size_t cardinality = fileSize / bytesPerRecord;

    if (cardinality > MAX_ELEMENT) {
        char message[1024];
        sprintf(message, "File size is %d (> max_elements %d)  [%s]",
            cardinality, MAX_ELEMENT, fvecs);
        throw std::runtime_error(message);
    }

    Tensor<T, 2, true> t({int(cardinality), dim});
    T* data = t.data();

    fin.read((char*)data, sizeof(T) * dim);
    for (int i = 1; i < cardinality; ++i) {
        fin.read((char*)&dim, 4);
        fin.read((char*)(data + i * dim),
                 sizeof(T) * dim);
    }
    fin.close();
    return t;
}

}
#endif //TENSOR_IO_H