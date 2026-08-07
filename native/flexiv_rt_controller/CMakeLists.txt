cmake_minimum_required(VERSION 3.16.3)

project(dfc_flexiv_rt_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)

# RDK 1.8's Python extension contains and dynamically exports the complete C++
# Robot implementation, including the RT methods omitted from pybind. Linking
# that already-installed, ABI-matched image avoids loading a second RDK runtime
# and avoids the fmt/spdlog ABI mismatch in the standalone static release.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import pathlib, flexivrdk, flexivrdk.flexivrdk as native; assert flexivrdk.__version__ == '1.8.0'; print(pathlib.Path(native.__file__).resolve())"
  RESULT_VARIABLE FLEXIVRDK_IMPORT_STATUS
  OUTPUT_VARIABLE FLEXIVRDK_EXTENSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT FLEXIVRDK_IMPORT_STATUS EQUAL 0 OR NOT EXISTS "${FLEXIVRDK_EXTENSION}")
  message(FATAL_ERROR
    "flexivrdk 1.8 must be installed in ${Python3_EXECUTABLE} before building")
endif()

set(FLEXIV_RDK_SOURCE_DIR "" CACHE PATH "Path to a flexiv_rdk v1.8 source checkout")
if(NOT FLEXIV_RDK_SOURCE_DIR)
  include(FetchContent)
  FetchContent_Declare(
    flexiv_rdk_headers
    GIT_REPOSITORY https://github.com/flexivrobotics/flexiv_rdk.git
    GIT_TAG a2dcedc637dc413b0e51c1b6442a614874d60f33
  )
  FetchContent_GetProperties(flexiv_rdk_headers)
  if(NOT flexiv_rdk_headers_POPULATED)
    FetchContent_Populate(flexiv_rdk_headers)
  endif()
  set(FLEXIV_RDK_SOURCE_DIR "${flexiv_rdk_headers_SOURCE_DIR}")
endif()
if(NOT EXISTS "${FLEXIV_RDK_SOURCE_DIR}/include/flexiv/rdk/robot.hpp")
  message(FATAL_ERROR "FLEXIV_RDK_SOURCE_DIR is not an RDK 1.8 source checkout")
endif()

add_executable(dfc-flexiv-rt-controller main.cpp)
add_library(flexivrdk_native SHARED IMPORTED)
set_target_properties(flexivrdk_native PROPERTIES
  IMPORTED_LOCATION "${FLEXIVRDK_EXTENSION}")
target_include_directories(dfc-flexiv-rt-controller PRIVATE
  "${FLEXIV_RDK_SOURCE_DIR}/include")
target_link_libraries(dfc-flexiv-rt-controller PRIVATE
  flexivrdk_native
  Python3::Python
  pthread
  rt
)
target_compile_options(dfc-flexiv-rt-controller PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS dfc-flexiv-rt-controller RUNTIME DESTINATION bin)
