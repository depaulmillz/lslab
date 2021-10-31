cmake_minimum_required(VERSION 3.18)

project(LSlab LANGUAGES CUDA CXX)

include(FetchContent)
include(cmake/config.cmake)

FetchContent_Declare(
        UnifiedMemoryGroupAllocation
        GIT_REPOSITORY git@github.com:sss-lehigh/UnifiedMemoryGroupAllocation.git
        GIT_TAG main
)

# CMake 3.14+
FetchContent_MakeAvailable(UnifiedMemoryGroupAllocation)

check_cuda_version(11.0 11.5)
set_consistent_standard(11)
setup_conan()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75")
    message(STATUS "Setting CUDA ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CUDA_FLAGS " -G -pg")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -pg")
endif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
string(APPEND CMAKE_CUDA_FLAGS " -Wreorder")

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

enable_testing()

add_library(lslab INTERFACE)
target_include_directories(lslab INTERFACE include)
target_link_libraries(lslab INTERFACE allocator)

add_executable(lslab_benchmark benchmark/benchmark.cu)
target_link_libraries(lslab_benchmark lslab)

add_executable(lslab_memory_benchmark benchmark/benchmark_mem.cu)
target_link_libraries(lslab_memory_benchmark lslab)

add_executable(slabunified_test test/slabunified_test.cu)
target_link_libraries(slabunified_test PRIVATE lslab CONAN_PKG::gtest ${CONAN_LIBS})
add_test(slabunified_test slabunified_test)

set(TARGETLIST lslab)

install(TARGETS ${TARGETLIST}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

set(CPACK_PACKAGE_VENDOR "dePaul Miller et. al.")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "dePaul Miller")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
        /.git
        /dist
        /.*build.*
        /\\\\.DS_Store
        )
include(CPack)
