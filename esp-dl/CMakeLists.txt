# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)

add_compile_options(-fdiagnostics-color=always)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(mamba_inference)

# If dataset.bin exists next to the model, include it in the flash target
# so that idf.py flash automatically flashes the dataset partition.
set(DATASET_BIN "${CMAKE_SOURCE_DIR}/main/model/dataset.bin")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${DATASET_BIN}")
if(EXISTS "${DATASET_BIN}")
    esptool_py_flash_to_partition(flash dataset "${DATASET_BIN}")
    message(STATUS "dataset.bin will be flashed to the dataset partition")
endif()
