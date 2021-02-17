project(darknet_tools)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH "${darknet_tools_SOURCE_DIR}/cmake")
message(" ** SOURCE DIR: " ${darknet_tools_SOURCE_DIR})

# Request C++11 standard, using new CMake variables.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# flags and parameters
set (WITH_GPU True CACHE BOOL "Use CUDA")
set (WITH_CUDNN True CACHE BOOL "Use libcudnn")
set (WITH_ADDRESS_SANITIZER false CACHE BOOL "Enable address sanitizer. NOTE: only works without cuda/cudnn")
set (default_build_type "Release")

# original darknet root location
set(DARKNET_ROOT "/mnt/mydata/tutorials/darknet_projects/darknet-cpp")
message(" ** Darknet root: " ${DARKNET_ROOT} )

# warning related flags
set (CMAKE_CXX_FLAGS "-Wall -Wextra")

# address sanitizer
set (WITH_ADDRESS_SANITIZER_STR "0")
if (WITH_ADDRESS_SANITIZER)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set (WITH_ADDRESS_SANITIZER_STR "1")
endif()

# GPU specifics
set (WITH_GPU_STR "0")
if (${WITH_GPU})
    set (WITH_GPU_STR "1")
    add_definitions(-DGPU)
    find_package(CUDA REQUIRED)
    include(select_compute_arch) # locate inside "cmake/" folder
    CUDA_SELECT_NVCC_ARCH_FLAGS(DETECTED_COMPUTE_ARCH_FLAGS)
    set (COMPUTE_ARCH_FLAGS ${DETECTED_COMPUTE_ARCH_FLAGS} CACHE STRING "Compute capability flags to compile for")
endif()

set (CUDNN_LIBS "")
set (WITH_CUDNN_STR "0")
if (${WITH_CUDNN})
    set (WITH_CUDNN_STR "1")
    add_definitions(-DCUDNN)
    set (CUDNN_LIBS "-lcudnn")
endif()

find_package(OpenCV 4 REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread signals filesystem)
find_package(Threads)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP yaml-cpp REQUIRED)
if(YAMLCPP_VERSION VERSION_GREATER "0.5.0")
    add_definitions(-DHAVE_YAMLCPP_GT_0_5_0)
endif()

find_package(yaml-cpp REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(
	include
	${darknet_tools_SOURCE_DIR}/include 
	${OpenCV_INCLUDE_DIRS}
	${YAMLCPP_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${DARKNET_ROOT}
	${DARKNET_ROOT}/include
	${CUDA_INCLUDE_DIRS}
)
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${YAMLCPP_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
link_directories (${DARKNET_ROOT}/src)

add_executable(intersection_over_union 
	src/intersection_over_union.cpp
	include/intersection_over_union/cvdnn_detector.cpp
	include/utils.cpp
)
target_link_libraries(intersection_over_union ${OpenCV_LIBRARIES} ${YAMLCPP_LIBRARIES} ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_REGEX_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
