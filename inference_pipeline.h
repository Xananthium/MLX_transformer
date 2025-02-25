#pragma once

#include <string>
#include <vector>
#include <functional>

#include "model_loader.h"
#include "transformer_model.h"

namespace mlx_transformer {

class InferencePipeline {
public:
    InferencePipeline(const std::string& model_path, const QuantizationOptions& quant_options = {});
    
    // Generate text given a prompt
    std::string generate(
        const std::string& prompt,
        int max_length = 100,
        float temperature = 0.7,
        int top_k = 50);
    
    // Streaming version of generate
    void generate_stream(
        const std::string& prompt,
        std::function<void(const std::string&)> token_callback,
        int max_length = 100,
        float temperature = 0.7,
        int top_k = 50);

private:
    ModelLoader loader_;
    TransformerModel model_;
    
    // Very simplified tokenizer for Phase 1
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& tokens);
};

// C API functions for external use
extern "C" {
    // Model management
    void* loadModel(const char* path, int quantization_mode);
    void unloadModel(void* model);
    
    // Text generation
    const char* generateText(void* model, const char* prompt, int max_length, float temperature, int top_k);
    void freeGeneratedText(const char* text);
}

} // namespace mlx_transformer