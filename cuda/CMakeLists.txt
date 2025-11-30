add_library(cuda_filters
    cuda_image_filters.cu
)

target_include_directories(cuda_filters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
