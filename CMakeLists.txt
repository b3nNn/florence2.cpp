cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(bart-cpp)

find_library(hf_tokenizers NAMES hf_tokenizers_c.dll PATHS "${CMAKE_SOURCE_DIR}/target/release")

#message("${CMAKE_CURRENT_SOURCE_DIR}/target/release")
#message("${hf_tokenizers}")

# Path to the Rust library
set(RUST_LIB_DIR "${CMAKE_SOURCE_DIR}/target/release") # Adjust this path to your Rust project
set(RUST_LIB_NAME "hf_tokenizers_c") # Name of the Rust library without prefix/suffix

# Path to the ggml library
set(GGML_LIB_DIR "${CMAKE_SOURCE_DIR}/llama.cpp/build/ggml/src/Release") # Adjust this path to your Rust project
set(GGML_LIB_NAME "ggml-base") # Name of the Rust library without prefix/suffix
set(GGML_DLL_DIR "${CMAKE_SOURCE_DIR}/llama.cpp/build/bin/Release") # Adjust this path to your Rust project
set(GGML_DLL_NAME "ggml-base") # Name of the Rust library without prefix/suffix

find_library(libggml NAMES ${GGML_LIB_NAME}.libbb PATHS ${GGML_LIB_DIR})
find_library(ggml NAMES ${GGML_DLL_NAME}.dll PATHS ${GGML_DLL_DIR})
find_library(ggml-cpu NAMES ${GGML_DLL_NAME}-cpu.dll PATHS ${GGML_DLL_DIR})

include_directories(./hf_tokenizer)
include_directories("$ENV{Torch_DIR}/include")
include_directories("$ENV{Torch_DIR}/include/torch/csrc/api/include")
set(RUST_LIB_DIR "${CMAKE_SOURCE_DIR}/target/release")
add_executable(bart-cpp test_florence2_gguf.cpp bart_tokenizer_fast.cpp hf_tokenizer/hf_tokenizers.cpp clip_image_processor.cpp florence2_processor.cpp)
target_include_directories(bart-cpp PRIVATE ${CMAKE_SOURCE_DIR}/hf_tokenizer)
set_property(TARGET bart-cpp PROPERTY CXX_STANDARD 17)
#
### On Windows, link against additional system libraries required by Rust
#if (WIN32)
    message("test4 [${${OpenCV_LIBS}}]")
    target_link_libraries(bart-cpp PRIVATE "${RUST_LIB_DIR}/${RUST_LIB_NAME}.dll.lib" "$ENV{Torch_LIB}/torch_cpu.lib" "$ENV{Torch_LIB}/c10.lib" opencv_core opencv_imgproc ws2_32 userenv bcrypt)
#else()
#    target_link_libraries(bart-cpp PRIVATE "${RUST_LIB_DIR}/hf_tokenizers_c.lib" opencv_core opencv_imgproc dl pthread)
#    message(WARNING "Only WIN32 platform is tested yet.")
#endif()
#
#target_include_directories(bart-cpp PRIVATE ${CMAKE_SOURCE_DIR}/hf_tokenizer)
#
## Ensure the Rust library is built before linking (optional, if Rust is part of the same build)
## If the Rust library is built separately, ensure cargo build --release has been run
#find_program(CARGO cargo)
#if (CARGO)
#    add_custom_target(rust_library
#            COMMAND ${CARGO} build --release
#            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#            COMMENT "Building Rust library"
#            )
#    add_dependencies(bart-cpp rust_library)
#endif()
#
find_package(OpenCV REQUIRED)
#target_include_directories(bart-cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(bart-cpp PRIVATE ${OpenCV_LIBS})
#
#find_package(Torch REQUIRED)
#message("test [$ENV{Torch_DIR}]")
#message("test2 [${OpenCV_INCLUDE_DIRS}]")
#message("test3 [${OpenCV_LIBS}]")

# On Windows, ensure the DLL is copied to the output directory
if(WIN32)
    file(GLOB TORCH_DLLS "$ENV{Torch_DIR}/lib/*.dll")
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RUST_LIB_DIR}/${RUST_LIB_NAME}.dll"
            $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GGML_DLL_DIR}/${GGML_DLL_NAME}.dll"
            $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$ENV{OpenCV_BIN}/opencv_world4110.dll"
            $<TARGET_FILE_DIR:bart-cpp>)
    # For Debug
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$ENV{OpenCV_BIN}/opencv_world4110d.dll"
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