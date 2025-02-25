#pragma once

#include <mlx/array.h>
#include <tuple>

namespace mlx_transformer {

enum class QuantizationMode {
    NONE,
    INT4,
    INT8
};

struct QuantizationOptions {
    QuantizationMode mode = QuantizationMode::NONE;
    bool use_zero_point = true;
    bool per_channel = true;
};

class Quantizer {
public:
    static mlx::core::array dequantize_int4(
        const mlx::core::array& quantized_weights,
        const mlx::core::array& scales,
        const mlx::core::array& zero_points = {});

    static std::tuple<mlx::core::array, mlx::core::array, mlx::core::array> quantize_int4(
        const mlx::core::array& weights,
        bool use_zero_point = true,
        bool per_channel = true);
};

} // namespace mlx_transformer