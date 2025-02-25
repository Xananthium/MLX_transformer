cmake_minimum_required(VERSION 3.12)
project(mlx_transformer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX package
find_package(mlx REQUIRED)

# Library sources
set(LIB_SOURCES
    memory_mapped_file.cpp
    quantizer.cpp
    model_loader.cpp
    attention.cpp
    feed_forward.cpp
    transformer_block.cpp
    transformer_model.cpp
    inference_pipeline.cpp
)

# Create the main library
add_library(mlx_transformer ${LIB_SOURCES})
target_include_directories(mlx_transformer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mlx_transformer PUBLIC mlx::mlx)

# Main executable
add_executable(transformer_example main.cpp)
target_link_libraries(transformer_example PRIVATE mlx_transformer)

# Installation
install(TARGETS mlx_transformer transformer_example
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Install headers
install(FILES 
    memory_mapped_file.h
    quantizer.h
    model_loader.h
    attention.h
    feed_forward.h
    transformer_block.h
    transformer_model.h
    inference_pipeline.h
    DESTINATION include/mlx_transformer)
