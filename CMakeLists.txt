# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.10)

# Project's name
project(Roberta)

# Set the C standard for the project
set(CMAKE_C_STANDARD 99)

add_library(Tensor STATIC Tensor.c)
target_include_directories(Tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add executable target with source files
add_executable(Roberta roberta.c Tensor.c)
add_compile_options(
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -mavx
        )
target_link_libraries(Roberta Tensor)
