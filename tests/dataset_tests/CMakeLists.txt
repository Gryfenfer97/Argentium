file(GLOB SRC
    "*.hpp"
    "*.cpp"
)

add_executable(dataset_tests ${SRC})

target_link_libraries(dataset_tests PRIVATE argentium gtest)
