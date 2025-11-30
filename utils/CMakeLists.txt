add_library(utils
    file_utils.cpp
    time_utils.cpp
)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
