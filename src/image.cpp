#include "image.hpp"
#include "stb/stb_image.h"
#include "stb/stb_image_resize.h"
#include <cstdio>

Image Image::open(const std::string& url) {
    Image img;
    
    // Validate input
    if (url.empty()) {
        img.m_has_error = true;
        img.m_error_message = "Empty file path provided";
        std::fprintf(stderr, "Error: Empty file path provided\n");
        return img;
    }

    std::fprintf(stdout, "Attempting to load image from: %s\n", url.c_str());
    
    // Load the image using stb_image
    int channels;
    unsigned char* data = stbi_load(url.c_str(), &img.m_width, &img.m_height, &channels, 0);
    
    if (data == nullptr) {
        img.m_has_error = true;
        img.m_error_message = "Failed to load image: " + std::string(stbi_failure_reason());
        std::fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        return img;
    }
    
    std::fprintf(stdout, "Successfully loaded image: %dx%d with %d channels\n", 
                img.m_width, img.m_height, channels);
    
    // Validate image dimensions
    if (img.m_width <= 0 || img.m_height <= 0) {
        img.m_has_error = true;
        img.m_error_message = "Invalid image dimensions";
        std::fprintf(stderr, "Error: Invalid image dimensions (%dx%d)\n", 
                    img.m_width, img.m_height);
        stbi_image_free(data);
        return img;
    }
    
    // Validate number of channels
    if (channels < 1 || channels > 4) {
        img.m_has_error = true;
        img.m_error_message = "Unsupported number of channels";
        std::fprintf(stderr, "Error: Unsupported number of channels (%d)\n", channels);
        stbi_image_free(data);
        return img;
    }
    
    // Store the image data
    size_t data_size = img.m_width * img.m_height * channels;
    img.m_data = std::vector<unsigned char>(data, data + data_size);
    img.m_channels = channels;
    
    std::fprintf(stdout, "Image data size: %zu bytes\n", data_size);
    
    // Free the stb_image allocated memory
    stbi_image_free(data);
    
    return img;
}

bool Image::resize(int new_width, int new_height) {
    // Validate input dimensions
    if (new_width <= 0 || new_height <= 0) {
        m_has_error = true;
        m_error_message = "Invalid dimensions for resize operation";
        std::fprintf(stderr, "Error: Invalid dimensions (%dx%d) for resize operation\n", 
                    new_width, new_height);
        return false;
    }

    // Check if image is empty or has error
    if (m_data.empty() || m_has_error) {
        m_has_error = true;
        m_error_message = "Cannot resize invalid or empty image";
        std::fprintf(stderr, "Error: Cannot resize invalid or empty image\n");
        return false;
    }

    std::fprintf(stdout, "Resizing image from %dx%d to %dx%d\n", 
                m_width, m_height, new_width, new_height);

    // Create temporary buffer for the resized image
    std::vector<unsigned char> resized_data(new_width * new_height * m_channels);
    
    // Perform the resize operation
    int result = stbir_resize_uint8(
        m_data.data(), m_width, m_height, 0,  // Source image
        resized_data.data(), new_width, new_height, 0,  // Destination image
        m_channels,  // Number of channels
        STBIR_FLAG_ALPHA_PREMULTIPLIED  // Alpha handling
    );

    if (result == 0) {
        m_has_error = true;
        m_error_message = "Failed to resize image";
        std::fprintf(stderr, "Error: Failed to resize image\n");
        return false;
    }

    // Update image data and dimensions
    m_data = std::move(resized_data);
    m_width = new_width;
    m_height = new_height;

    std::fprintf(stdout, "Successfully resized image to %dx%d\n", new_width, new_height);
    return true;
}

int Image::width() const {
    return m_width;
}

int Image::height() const {
    return m_height;
}

bool Image::has_error() const {
    return m_has_error;
}

std::string Image::error_message() const {
    return m_error_message;
}

int Image::channels() const {
    return m_channels;
}

const std::vector<unsigned char>& Image::data() const {
    return m_data;
}