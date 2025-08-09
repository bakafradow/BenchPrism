cmake_minimum_required(VERSION 3.20)
project(CppGTestRunner)

find_package(GTest REQUIRED)
include_directories(.)
add_executable(test test.cpp)
target_link_libraries(test PRIVATE GTest::gtest_main)
