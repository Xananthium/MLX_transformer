#pragma once

#include <string>
#include <unordered_map>
#include <mlx/array.h>

#include "quantizer.h"

namespace mlx_transformer {

struct ModelConfig {
    int64_t vocab_size;
    int64_t hidden_size;
    int64_t intermediate_size;
    int64_t num_hidden_layers;
    int64_t num_attention_heads;
    int64_t max_position_embeddings;
    float layer_norm_epsilon;
    std::string model_type;
};

class ModelLoader {
public:
    ModelLoader(const std::string& model_path, const QuantizationOptions& quant_options = {});
    
    const ModelConfig& config() const;
    
    // Lazy loading of weights - only loads when requested
    mlx::core::array loadWeight(const std::string& name);
    
    // Preload common weights to improve initial inference time
    void preloadCommonWeights();
    
    // Clear the weight cache to free memory
    void clearWeightCache();

private:
    std::string model_path_;
    QuantizationOptions quant_options_;
    ModelConfig config_;
    std::unordered_map<std::string, mlx::core::array> weight_cache_;
    
    void loadConfig();
};

} // namespace mlx_transformer