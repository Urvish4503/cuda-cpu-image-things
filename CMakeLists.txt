cmake_minimum_required(VERSION 3.18)
project(ImageProcessor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# Enable useful CUDA features
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75)   # your 1650Ti, change if needed

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add folders
add_subdirectory(utils)
add_subdirectory(cpu)
add_subdirectory(cuda)

# Main executable
add_executable(image_processor main.cpp)


# Link everything
target_include_directories(image_processor
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(image_processor
    PRIVATE
        utils
        cpu_filters
        cuda_filters
        ${OpenCV_LIBS}
)