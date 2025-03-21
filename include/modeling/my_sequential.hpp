#ifndef MYSEQUENTIAL_H
#define MYSEQUENTIAL_H

#include <torch/torch.h>

class MySequential : public torch::nn::Module {
public:
    torch::nn::ModuleList modules;

    MySequential() : modules() {}

    template <typename... Modules>
    MySequential(Modules... modules_) : modules{modules_...} {}

    torch::Tensor forward(torch::Tensor x);
};

#endif // MYSEQUENTIAL_H