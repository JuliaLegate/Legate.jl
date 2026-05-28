cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(LegateWrapper)

set(LegateWrapperVersion 0.0.1)
message(STATUS "Project version: v${LegateWrapperVersion}")
set(LIBRARY_NAME legate_jl_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(LEGATE_WRAPPER_ENABLE_CUDA "Build legate_jl_wrapper with CUDA support" ON)
option(BINARYBUILDER "Building with binary builder" ON)

if(LEGATE_WRAPPER_ENABLE_CUDA)
    find_package(CUDAToolkit 13.0 REQUIRED)
else()
    # only disables find_package requirement for CUDAToolkit.
    # if you have a CUDA enabled Legate install, this really won't do anything.
    message(STATUS "LEGATE_WRAPPER_ENABLE_CUDA=OFF: skipping CUDAToolkit in legate_jl_wrapper.")
endif()

find_package(legate REQUIRED)

# CxxWrap Stuff
if(NOT BINARYBUILDER)
execute_process(
  COMMAND julia -e "println(DEPOT_PATH[1])"
  OUTPUT_VARIABLE JULIA_DEP_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(JlCxx_DIR "${JULIA_DEP_PATH}/dev/libcxxwrap_julia_jll/override")
message(STATUS "Setting JlCxx_DIR to ${JlCxx_DIR} (not using BinaryBuilder)")
endif()

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

set(SOURCES
    src/types.cpp
    src/module.cpp
    src/task.cpp
)

add_library(${LIBRARY_NAME} SHARED ${SOURCES})
set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${LegateWrapperVersion})

target_link_libraries(${LIBRARY_NAME} PRIVATE legate::legate JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl)
target_include_directories(${LIBRARY_NAME} PRIVATE include)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
