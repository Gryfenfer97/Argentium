include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


add_subdirectory(dataset_tests)
add_subdirectory(activation_test)

