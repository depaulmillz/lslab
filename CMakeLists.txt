cmake_minimum_required(VERSION 3.18)

project(LSlab LANGUAGES CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "SET CMAKE_BUILD_TYPE to Release")
endif()

include(cmake/config.cmake)

check_cuda_version(11.0 11.6)
set_consistent_standard(17)

find_package(UnifiedMemoryGroupAllocation REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
    message(STATUS "Setting CUDA ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES}")
endif()

message(STATUS "Using CUDA ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -pg")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -pg")
endif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
string(APPEND CMAKE_CUDA_FLAGS " -Wreorder")

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

# disable async calls on UM for Maxwell, which can break code
foreach(ARCH "${CMAKE_CUDA_ARCHITECTURES}")
    if("${ARCH}" MATCHES "^(5.)$")
        add_compile_options("-DDISABLE_PREFETCH")
        break()
    endif()
endforeach()

enable_testing()

add_library(lslab INTERFACE)
target_include_directories(lslab INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(lslab INTERFACE UnifiedMemoryGroupAllocation::UnifiedMemoryGroupAllocation)

add_executable(lslab_benchmark benchmark/benchmark.cu)
target_link_libraries(lslab_benchmark lslab)
#
#add_executable(lslab_memory_benchmark benchmark/benchmark_mem.cu)
#target_link_libraries(lslab_memory_benchmark lslab)

add_executable(slabunified_test test/slabunified_test.cu)
target_link_libraries(slabunified_test PRIVATE lslab)
add_test(NAME slabunified_test COMMAND slabunified_test)

#add_executable(asanslabunified_test test/slabunified_test.cu)
#target_link_libraries(asanslabunified_test PRIVATE asan lslab gcov)
#target_compile_options(asanslabunified_test PRIVATE -Xcompiler '--sanitize=address') 

set(TARGETLIST lslab)

install(TARGETS ${TARGETLIST}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)
#
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

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
