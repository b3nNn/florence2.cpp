#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION

class Image {
public:
    Image() = default;

    // Constructor to load image from a URL
    static Image open(const std::string& url);

    // Getter for image width
    int width() const;

    // Getter for image height
    int height() const;

    // Error handling
    bool has_error() const;
    std::string error_message() const;

private:
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    std::vector<unsigned char> m_data;
    bool m_has_error = false;
    std::string m_error_message;
};

#endif // IMAGE_HPP