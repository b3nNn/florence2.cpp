cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(bart-cpp)

# Rust library
find_library(hf_tokenizers NAMES hf_tokenizers_c.dll PATHS "${CMAKE_SOURCE_DIR}/target/release")
set(RUST_LIB_DIR "${CMAKE_SOURCE_DIR}/target/release")
set(RUST_LIB_NAME "hf_tokenizers_c")

# GGML library (unused in Florence-2, kept for completeness)
set(GGML_LIB_DIR "${CMAKE_SOURCE_DIR}/llama.cpp/build/ggml/src/Release")
set(GGML_DLL_DIR "${CMAKE_SOURCE_DIR}/llama.cpp/build/bin/Release")
set(GGML_DLL_NAME "ggml-base")

# ONNX Runtime
set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_LIB}/lib)

# Find libprotobuf.lib
find_library(PROTOBUF_LIB
        NAMES libprotobuf protobuf
        PATHS "${ONNXRUNTIME_LIB}" "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/build/Release"
        DOC "Path to libprotobuf.lib"
        )
if(NOT PROTOBUF_LIB)
    message(FATAL_ERROR "Could not find libprotobuf.lib. Ensure it’s in ${ONNXRUNTIME_LIB} or ${ONNXRUNTIME_ROOT}/lib.")
endif()
message(STATUS "Found libprotobuf.lib at: ${PROTOBUF_LIB}")

## Find Abseil libraries
#set(ABSL_LIBS_NAMES
#        absl_hash
#        absl_base
#        absl_strings
#        absl_raw_hash_set
#        absl_synchronization  # Common Abseil libs needed by ONNX Runtime
#        )
#foreach(lib ${ABSL_LIBS_NAMES})
#    find_library(${lib}_LIB
#            NAMES ${lib}
#            PATHS "${ONNXRUNTIME_LIB}" "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/build/Release"
#            DOC "Path to ${lib}.lib"
#            )
#    if(NOT ${lib}_LIB)
#        message(WARNING "Could not find ${lib}.lib. This may cause linker errors.")
#    else()
#        message(STATUS "Found ${lib}.lib at: ${${lib}_LIB}")
#        list(APPEND ABSL_LIBS ${${lib}_LIB})
#    endif()
#endforeach()

# Torch
set(Torch_DIR $ENV{Torch_DIR})
set(Torch_LIB $ENV{Torch_LIB})
include_directories("${Torch_DIR}/include" "${Torch_DIR}/include/torch/csrc/api/include")

# OpenCV
find_package(OpenCV REQUIRED)

# Executable
add_executable(bart-cpp
        test_florence2_onnx.cpp
        bart_tokenizer_fast.cpp
        hf_tokenizer/hf_tokenizers.cpp
        clip_image_processor.cpp
        florence2_processor.cpp
        )
target_include_directories(bart-cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/hf_tokenizer
        ${CMAKE_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        )
set_property(TARGET bart-cpp PROPERTY CXX_STANDARD 17)

# Link libraries
if(WIN32)
    target_link_libraries(bart-cpp PRIVATE
            "${RUST_LIB_DIR}/${RUST_LIB_NAME}.dll.lib"
            "${Torch_LIB}/torch.lib"
            "${Torch_LIB}/torch_cpu.lib"
            "${Torch_LIB}/c10.lib"
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib"
            ${PROTOBUF_LIB}
            ${ABSL_LIBS}  # Link all found Abseil libraries
            ${OpenCV_LIBS}
            ws2_32 userenv bcrypt
            )
else()
    message(WARNING "Only WIN32 platform is tested yet.")
endif()

# Copy DLLs and assets
if(WIN32)
    file(GLOB TORCH_DLLS "${Torch_LIB}/*.dll")
    file(GLOB ONNX_DLLS "${ONNXRUNTIME_ROOT}/*.dll")
    add_custom_command(TARGET bart-cpp PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RUST_LIB_DIR}/${RUST_LIB_NAME}.dll" $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GGML_DLL_DIR}/${GGML_DLL_NAME}.dll" $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "$ENV{OpenCV_BIN}/opencv_world4110.dll" $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "$ENV{OpenCV_BIN}/opencv_world4110d.dll" $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" $<TARGET_FILE_DIR:bart-cpp>)
    add_custom_command(TARGET bart-cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll" $<TARGET_FILE_DIR:bart-cpp>)
endif()

add_custom_command(TARGET bart-cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/tokenizer.json" $<TARGET_FILE_DIR:bart-cpp>)
add_custom_command(TARGET bart-cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/config.json" $<TARGET_FILE_DIR:bart-cpp>)
add_custom_command(TARGET bart-cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/puma.png" $<TARGET_FILE_DIR:bart-cpp>)
add_custom_command(TARGET bart-cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/florence2.onnx" $<TARGET_FILE_DIR:bart-cpp>)