cmake_minimum_required(VERSION 2.8)
project(gridop2)
set(CMAKE_VERBOSE_MAKEFILE off)

find_package(VTK REQUIRED)

include(${VTK_USE_FILE})

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET (CMAKE_CXX_FLAGS "-fopenmp -Wno-deprecated-declarations")
MESSAGE("USING FLAGS ${CMAKE_CXX_FLAGS}")
MESSAGE("COMPILER VERSION ${CMAKE_CXX_COMPILER_ID}")
MESSAGE("COMPILER VERSION ${CMAKE_CXX_COMPILER_VERSION}")

SET(SRC_LIST gridop2.cpp)

include_directories(${BOOST_ROOT})
include_directories("../../../include")
include_directories("../../gridop2_src")
include_directories("/usr/include/vtk-7.1")
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES}) 
