#include "MySequential.hpp"

torch::Tensor MySequential::forward(torch::Tensor x) {
    for (auto& module : modules) {
        x = (*module)(x);
    }
    return x;
}