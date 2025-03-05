cmake_minimum_required(VERSION 3.22.1)
project(cpu_dgemm VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_FORMAT "%Y-%m-%d %H:%M:%S")
set(CMAKE_C_FORMAT_INDENT "4")

include_directories(include)
find_package(BLAS REQUIRED)
add_executable(cpu_dgemm src/main.c src/my_dgemm.c src/utils.c src/kernels.c)
target_link_libraries(cpu_dgemm ${BLAS_LIBRARIES})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)