cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(bart-cpp)

find_library(hf_tokenizers NAMES hf_tokenizers_c.dll PATHS "${CMAKE_SOURCE_DIR}/target/release")

message("${CMAKE_CURRENT_SOURCE_DIR}/target/release")
message("${hf_tokenizers}")

# Path to the Rust library
set(RUST_LIB_DIR "${CMAKE_SOURCE_DIR}/target/release") # Adjust this path to your Rust project
set(RUST_LIB_NAME "hf_tokenizers_c") # Name of the Rust library without prefix/suffix

#include_directories(.)
include_directories(./hf_tokenizer)
set(RUST_LIB_DIR "${CMAKE_SOURCE_DIR}/target/release")
add_executable(bart-cpp test_florence2.cpp bart_tokenizer_fast.cpp hf_tokenizer/hf_tokenizers.cpp clip_image_processor.cpp florence2_processor.cpp)
set_property(TARGET bart-cpp PROPERTY CXX_STANDARD 17)

## On Windows, link against additional system libraries required by Rust
if (WIN32)
    target_link_libraries(bart-cpp PRIVATE "${RUST_LIB_DIR}/hf_tokenizers_c.dll.lib" ws2_32 userenv bcrypt)
else()
    target_link_libraries(bart-cpp PRIVATE "${RUST_LIB_DIR}/hf_tokenizers_c" dl pthread)
    message(WARNING "Only WIN32 platform is tested yet.")
endif()

target_include_directories(bart-cpp PRIVATE ${CMAKE_SOURCE_DIR}/hf_tokenizer ${CMAKE_SOURCE_DIR})

# Ensure the Rust library is built before linking (optional, if Rust is part of the same build)
# If the Rust library is built separately, ensure cargo build --release has been run
find_program(CARGO cargo)
if (CARGO)
    add_custom_target(rust_library
            COMMAND ${CARGO} build --release
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Building Rust library"
            )
    add_dependencies(bart-cpp rust_library)
endif()

find_package(OpenCV REQUIRED)
target_include_directories(bart-cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bart-cpp PRIVATE ${OpenCV_LIBS})

# On Windows, ensure the DLL is copied to the output directory
if(WIN32)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RUST_LIB_DIR}/${RUST_LIB_NAME}.dll"
            $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$ENV{OpenCV_BIN}/opencv_world4110.dll"
            $<TARGET_FILE_DIR:bart-cpp>)
endif()

add_custom_command(TARGET bart-cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/tokenizer.json"
        $<TARGET_FILE_DIR:bart-cpp>)

add_custom_command(TARGET bart-cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/puma.png"
        $<TARGET_FILE_DIR:bart-cpp>)