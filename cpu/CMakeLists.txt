add_library(cpu_filters
    image_filters.cpp
)

target_include_directories(cpu_filters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
