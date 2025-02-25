#pragma once

#include <string>
#include <stdexcept>

namespace mlx_transformer {

class MemoryMappedFile {
public:
    MemoryMappedFile(const std::string& path);
    ~MemoryMappedFile();

    void* data() const;
    size_t size() const;

    // Delete copy constructors to prevent accidental copies
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

private:
    int fd_;
    void* data_;
    size_t size_;
};

} // namespace mlx_transformer