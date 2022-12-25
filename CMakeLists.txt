cmake_minimum_required(VERSION 3.16)
project(Argentium)

option(ENABLE_BUILD_TEST "Build the tests" On)

add_subdirectory(src)

if(ENABLE_BUILD_TEST)
    add_subdirectory(tests)
endif()



