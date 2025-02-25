#include <iostream>
#include <string>

#include "inference_pipeline.h"

// Example function to print tokens as they're generated
void printToken(const std::string& token) {
    std::cout << token << std::flush;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [quantization_mode]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    int quantization_mode = 0;  // Default to no quantization
    
    if (argc >= 3) {
        quantization_mode = std::stoi(argv[2]);
    }
    
    try {
        // Set up quantization options
        mlx_transformer::QuantizationOptions quant_options;
        quant_options.mode = static_cast<mlx_transformer::QuantizationMode>(quantization_mode);
        
        std::cout << "Loading model from: " << model_path << std::endl;
        
        // Create inference pipeline
        mlx_transformer::InferencePipeline pipeline(model_path, quant_options);
        
        // Example 1: Basic text generation
        std::string prompt = "Once upon a time in a galaxy far, far away";
        std::cout << "\nGenerating text with prompt: " << prompt << std::endl;
        
        std::string generated_text = pipeline.generate(
            prompt,
            100,    // max_length
            0.7,    // temperature
            50      // top_k
        );
        
        std::cout << "Generated: " << generated_text << std::endl;
        
        // Example 2: Streaming generation
        std::cout << "\nStreaming generation with the same prompt:" << std::endl;
        std::cout << prompt;  // Print the prompt first
        
        pipeline.generate_stream(
            prompt,
            printToken,  // Callback function to process each token
            100,         // max_length
            0.7,         // temperature
            50           // top_k
        );
        
        std::cout << "\n\nGeneration complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}