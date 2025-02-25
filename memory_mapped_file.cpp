#include "memory_mapped_file.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace mlx_transformer {

MemoryMappedFile::MemoryMappedFile(const std::string& path) : fd_(-1), data_(nullptr), size_(0) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // Get file size
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw std::runtime_error("Failed to get file size: " + path);
    }
    size_ = sb.st_size;

    // Memory map the file
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Failed to memory map file: " + path);
    }

    // Advise the kernel that we'll access the data sequentially
    madvise(data_, size_, MADV_SEQUENTIAL);
}

MemoryMappedFile::~MemoryMappedFile() {
    if (data_ != nullptr && data_ != MAP_FAILED) {
        munmap(data_, size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

void* MemoryMappedFile::data() const {
    return data_;
}

size_t MemoryMappedFile::size() const {
    return size_;
}

} // namespace mlx_transformer