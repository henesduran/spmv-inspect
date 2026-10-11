cmake_minimum_required(VERSION 3.20)
project(spmv-inspect CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(spmv-inspect src/main.cpp)
target_include_directories(spmv-inspect PRIVATE include)
target_link_libraries(spmv-inspect PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(spmv-inspect PRIVATE -O3 -Wall -Wextra)