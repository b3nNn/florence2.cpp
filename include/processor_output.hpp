#ifndef PROCESSOR_OUTPUT_HPP
#define PROCESSOR_OUTPUT_HPP

#include <vector>
#include <optional>
#include <cstdint>
#include "image.hpp"

struct ProcessorOutput {
    const std::vector<uint32_t> input_ids;
    const std::vector<uint32_t> pixel_values;

    ProcessorOutput(
        const std::vector<uint32_t>& ids = {},
        const std::vector<uint32_t>& values = {}
    ) : input_ids(ids), pixel_values(values) {}
};

#endif // PROCESSOR_OUTPUT_HPP