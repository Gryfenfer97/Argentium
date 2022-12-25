file(GLOB SRC
    "*.hpp"
    "*.cpp"
)

add_executable(activation_tests ${SRC})

target_link_libraries(activation_tests PRIVATE argentium gtest)