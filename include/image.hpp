#ifndef IMAGE_HPP
#define IMAGE_HPP

/*
 * stb_image - v2.28
 * Public domain image loading library
 * Original source: https://github.com/nothings/stb
 * 
 * This is a single-file library that provides image loading functionality.
 * It supports PNG, JPEG, BMP, GIF, TGA, and HDR formats.
 * 
 * By Sean Barrett and contributors
 * License: Public Domain
 */

#include <string>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

class Image {
public:
    // Default constructor
    Image() = default;

    // Constructor to load image from a URL
    static Image open(const std::string& url);

    // Getters
    int width() const;
    int height() const;
    int channels() const;
    const std::vector<unsigned char>& data() const;
    bool has_error() const;
    std::string error_message() const;

    // Image operations
    bool resize(int new_width, int new_height);

private:
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    std::vector<unsigned char> m_data;
    bool m_has_error = false;
    std::string m_error_message;
};

#endif // IMAGE_HPP