cmake_minimum_required(VERSION 3.10)
project(Florence2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/image.cpp
    src/model_processor.cpp
    main.cpp
)

# Create executable
add_executable(florence2 ${SOURCES})