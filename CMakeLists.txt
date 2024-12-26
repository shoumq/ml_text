cmake_minimum_required(VERSION 3.29)
project(mltext)

set(CMAKE_CXX_STANDARD 23)

include_directories(crow)

add_executable(mltext main.cpp
        LogisticRegressor.cpp
        LogisticRegressor.h
        GradientBoosting.cpp
        GradientBoosting.h)
