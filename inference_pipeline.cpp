#include "inference_pipeline.h"

#include <iostream>
#include <cstring>

namespace mlx_transformer {

InferencePipeline::InferencePipeline(const std::string& model_path, const QuantizationOptions& quant_options)
    : loader_(model_path, quant_options),
      model_(loader_.config()) {
    
    // Load model weights
    model_.loadWeights(loader_);
    
    // Initialize tokenizer (simplified for Phase 1)
    // In a real implementation, this would load the tokenizer configuration
}

std::string InferencePipeline::generate(
    const std::string& prompt,
    int max_length,
    float temperature,
    int top_k) {
    
    // Tokenize input (simplified)
    auto input_ids = tokenize(prompt);
    
    // Clear KV cache before generation
    model_.clearKVCache();
    
    // Generate tokens
    std::vector<int> output_ids(input_ids.begin(), input_ids.end());
    
    for (int i = 0; i < max_length; i++) {
        // Convert to MLX array
        auto input_array = mlx::core::array(output_ids, mlx::core::int32);
        input_array = mlx::core::reshape(input_array, {1, -1});  // Add batch dimension
        
        // Generate next token
        auto next_token = model_.generate_next_token(input_array, temperature, top_k);
        
        // Convert to scalar and add to output
        int token_id = static_cast<int>(mlx::core::item<int>(next_token));
        output_ids.push_back(token_id);
        
        // Check for end of sequence token (simplified)
        if (token_id == 2) {  // Assuming 2 is EOS token
            break;
        }
    }
    
    // Detokenize output (simplified)
    return detokenize(output_ids);
}

void InferencePipeline::generate_stream(
    const std::string& prompt,
    std::function<void(const std::string&)> token_callback,
    int max_length,
    float temperature,
    int top_k) {
    
    // Tokenize input (simplified)
    auto input_ids = tokenize(prompt);
    
    // Clear KV cache before generation
    model_.clearKVCache();
    
    // Generate tokens
    std::vector<int> output_ids(input_ids.begin(), input_ids.end());
    
    for (int i = 0; i < max_length; i++) {
        // Convert to MLX array
        auto input_array = mlx::core::array({output_ids.back()}, mlx::core::int32);
        input_array = mlx::core::reshape(input_array, {1, 1});  // Add batch dimension
        
        // Generate next token
        auto next_token = model_.generate_next_token(input_array, temperature, top_k);
        
        // Convert to scalar and add to output
        int token_id = static_cast<int>(mlx::core::item<int>(next_token));
        output_ids.push_back(token_id);
        
        // Detokenize and send to callback
        std::string token_text = detokenize({token_id});
        token_callback(token_text);
        
        // Check for end of sequence token (simplified)
        if (token_id == 2) {  // Assuming 2 is EOS token
            break;
        }
    }
}

std::vector<int> InferencePipeline::tokenize(const std::string& text) {
    // Dummy implementation that just converts characters to integers
    std::vector<int> tokens;
    for (char c : text) {
        tokens.push_back(static_cast<int>(c));
    }
    return tokens;
}

std::string InferencePipeline::detokenize(const std::vector<int>& tokens) {
    // Dummy implementation that just converts integers to characters
    std::string text;
    for (int token : tokens) {
        if (token < 256) {  // Only convert ASCII range
            text.push_back(static_cast<char>(token));
        }
    }
    return text;
}

// C API implementations
extern "C" {

void* loadModel(const char* path, int quantization_mode) {
    try {
        QuantizationOptions options;
        options.mode = static_cast<QuantizationMode>(quantization_mode);
        
        auto* pipeline = new InferencePipeline(path, options);
        return static_cast<void*>(pipeline);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return nullptr;
    }
}

void unloadModel(void* model) {
    if (model) {
        auto* pipeline = static_cast<InferencePipeline*>(model);
        delete pipeline;
    }
}

const char* generateText(void* model, const char* prompt, int max_length, float temperature, int top_k) {
    if (!model || !prompt) {
        return nullptr;
    }
    
    try {
        auto* pipeline = static_cast<InferencePipeline*>(model);
        std::string result = pipeline->generate(prompt, max_length, temperature, top_k);
        
        // Allocate memory for the result (caller must free)
        char* output = new char[result.size() + 1];
        std::strcpy(output, result.c_str());
        return output;
    } catch (const std::exception& e) {
        std::cerr << "Error generating text: " << e.what() << std::endl;
        return nullptr;
    }
}

void freeGeneratedText(const char* text) {
    if (text) {
        delete[] text;
    }
}

} // extern "C"

} // namespace mlx_transformer