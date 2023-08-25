project(ch3 VERSION 0.1.0 LANGUAGES CXX)

find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS MPI)


foreach(target pi matvec matmat monte-carlo)
    add_executable(${target} ${target}.cc)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    target_link_libraries(${target} PRIVATE spdlog::spdlog_header_only)
    target_link_libraries(${target} PRIVATE fmt::fmt-header-only)
endforeach()

target_link_libraries(monte-carlo PRIVATE Boost::mpi)
# include(FindBoost)
include(FindMPI)
