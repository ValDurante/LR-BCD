cmake_minimum_required(VERSION 3.12)

project(LRBCD VERSION 1.0 LANGUAGES CXX)

include(CTest)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

file(GLOB lrbcd_src CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/src/*.hh"
)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(lrbcd ${lrbcd_src})

target_link_libraries(lrbcd PRIVATE Eigen3::Eigen)
