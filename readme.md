# MLX Transformer Library

This library provides a transformer-based language model implementation using the MLX framework. It supports efficient inference with memory-mapped weights, quantization, and streaming text generation.

## Project Structure

The project is structured into the following components:

- **memory_mapped_file**: Efficiently loads model weights using memory mapping
- **quantizer**: Provides int4 quantization support for model weights
- **model_loader**: Handles lazy loading of model weights from disk
- **attention**: Implements multi-head attention mechanism
- **feed_forward**: Implements the feed-forward network in transformer blocks
- **transformer_block**: Combines attention and feed-forward networks into a transformer layer
- **transformer_model**: Ties together the transformer layers to build the full model
- **inference_pipeline**: Provides a high-level API for text generation

## Building the Project

### Prerequisites

- CMake 3.12 or higher
- A C++17 compatible compiler
- MLX library installed

### Build Instructions

1. Clone the repository
2. Make the build script executable:
   ```
   chmod +x build.sh
   ```
3. Run the build script:
   ```
   ./build.sh
   ```

## Using the Library

### Example Usage

The `main.cpp` file demonstrates how to use the library to load a model and generate text:

```cpp
// Create an inference pipeline from a model path
mlx_transformer::InferencePipeline pipeline(model_path, quant_options);

// Generate text with a prompt
std::string generated_text = pipeline.generate(
    "Once upon a time in a galaxy far, far away",
    100,    // max_length
    0.7,    // temperature
    50      // top_k
);

// Stream generation with a callback
pipeline.generate_stream(
    prompt,
    printToken,  // Callback function to process each token
    100,         // max_length
    0.7,         // temperature
    50           // top_k
);
```

### Running the Example

After building, run the example with:

```
./build/transformer_example <model_path> [quantization_mode]
```

Where:
- `model_path` is the path to the model directory containing the weights
- `quantization_mode` is optional (0=None, 1=INT4, 2=INT8)

## C API

The library also provides a C API for use in other languages:

```c
// Load a model
void* model = loadModel("path/to/model", 0);

// Generate text
const char* text = generateText(model, "Hello, world!", 100, 0.7, 50);

// Free resources
freeGeneratedText(text);
unloadModel(model);
```
