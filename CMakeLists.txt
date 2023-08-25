cmake_minimum_required(VERSION 3.21.0)
project(using_mpi VERSION 0.1.0 LANGUAGES CXX)

set(CXX_STANDARD_REQUIRED ON)
set(CXX_STANDARD 11)

add_subdirectory(ch3)
include(CTest)
enable_testing()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
