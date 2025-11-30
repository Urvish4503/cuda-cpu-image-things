#pragma once
#include <chrono>

struct Timer
{
    std::chrono::high_resolution_clock::time_point start;

    Timer() { reset(); }

    void reset()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    double ms() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start).count();
    }
};
