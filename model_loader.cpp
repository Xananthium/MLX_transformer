#include "model_loader.h"

#include <mlx/io.h>
#include <filesystem>
#include <iostream>

#include "memory_mapped_file.h"

namespace mlx_transformer {

ModelLoader::ModelLoader(const std::string& model_path, const QuantizationOptions& quant_options)
    : model_path_(model_path), quant_options_(quant_options) {
    // Ensure the model path exists
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model path does not exist: " + model_path);
    }
    
    // Load model configuration
    loadConfig();
}

const ModelConfig& ModelLoader::config() const {
    return config_;
}

mlx::core::array ModelLoader::loadWeight(const std::string& name) {
    auto it = weight_cache_.find(name);
    if (it != weight_cache_.end()) {
        return it->second;
    }
    
    // Load the weight from disk
    std::string weight_path = model_path_ + "/weights/" + name + ".safetensors";
    if (!std::filesystem::exists(weight_path)) {
        throw std::runtime_error("Weight file not found: " + weight_path);
    }
    
    // Memory map the file
    MemoryMappedFile mmapped_file(weight_path);
    
    // Parse safetensors format and extract array
    // This is a simplified implementation for Phase 1
    mlx::core::array weight = mlx::io::load_safetensors(weight_path)[name];
    
    // Apply quantization if needed
    if (quant_options_.mode == QuantizationMode::INT4) {
        // For phase 1, we're just storing the weights as-is
        // In a real implementation, weights would be loaded in quantized format
        // and dequantized on demand
        weight_cache_[name] = weight;
    } else {
        weight_cache_[name] = weight;
    }
    
    return weight_cache_[name];
}

void ModelLoader::preloadCommonWeights() {
    std::vector<std::string> common_weights = {
        "embedding.weight",
        "lm_head.weight"
    };
    
    for (int i = 0; i < std::min(2, config_.num_hidden_layers); i++) {
        common_weights.push_back("transformer.layers." + std::to_string(i) + ".attention.wq.weight");
        common_weights.push_back("transformer.layers." + std::to_string(i) + ".attention.wk.weight");
        common_weights.push_back("transformer.layers." + std::to_string(i) + ".attention.wv.weight");
        common_weights.push_back("transformer.layers." + std::to_string(i) + ".attention.wo.weight");
    }
    
    for (const auto& name : common_weights) {
        try {
            loadWeight(name);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to preload weight " << name << ": " << e.what() << std::endl;
        }
    }
}

void ModelLoader::clearWeightCache() {
    weight_cache_.clear();
}

void ModelLoader::loadConfig() {
    std::string config_path = model_path_ + "/config.json";
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("Model configuration file not found: " + config_path);
    }
    
    // In a real implementation, parse the JSON file
    // For Phase 1, we'll use a simplified approach with hardcoded defaults
    config_.vocab_size = 32000;
    config_.hidden_size = 4096;
    config_.intermediate_size = 11008;
    config_.num_hidden_layers = 32;
    config_.num_attention_heads = 32;
    config_.max_position_embeddings = 4096;
    config_.layer_norm_epsilon = 1e-5;
    config_.model_type = "llama";
    
    // In a real implementation, read these values from the config file
}

} // namespace mlx_transformer