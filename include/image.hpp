#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>

class Image {
public:
    Image() = default;

    // Constructor to load image from a URL
    static Image open(const std::string& url);

    // Getter for image width
    int width() const;

    // Getter for image height
    int height() const;
};

#endif // IMAGE_HPP