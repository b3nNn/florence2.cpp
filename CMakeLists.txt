cmake_minimum_required(VERSION 3.10)
project(Florence2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Rust library
set(RUST_LIB_DIR "${CMAKE_SOURCE_DIR}/hf_tokenizer")
set(RUST_LIB_NAME "hf_tokenizer")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include ${RUST_LIB_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)

# Source files
set(SOURCES
    src/image.cpp
    src/model_processor.cpp
    src/tokenizer.cpp
    main.cpp
)

# Create executable
add_executable(florence2 ${SOURCES})

# Ensure the Rust library is built before linking (optional, if Rust is part of the same build)
# If the Rust library is built separately, ensure cargo build --release has been run
find_program(CARGO cargo)
if (CARGO)
    add_custom_target(hf_tokenizer
            COMMAND ${CARGO} build --release
            WORKING_DIRECTORY ${RUST_LIB_DIR}
            COMMENT "Building Rust library"
            )
    add_dependencies(florence2 hf_tokenizer)
endif()

# Copy DLLs and assets
if(WIN32)
    add_custom_command(TARGET florence2 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RUST_LIB_DIR}/target/release/${RUST_LIB_NAME}.dll" $<TARGET_FILE_DIR:florence2>)
endif()

# Link libraries
if(WIN32)
    target_link_libraries(florence2 PRIVATE
            "${RUST_LIB_DIR}/target/release/${RUST_LIB_NAME}.dll.lib"
            ws2_32 userenv bcrypt
            )
else()
    target_link_libraries(florence2 PRIVATE
            "${RUST_LIB_DIR}/target/release/${RUST_LIB_NAME}.lib"
            )
endif()