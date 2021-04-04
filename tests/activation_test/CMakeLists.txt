file(GLOB SRC
    "*.hpp"
    "*.cpp"
)

include_directories(.. tests)

add_executable(ActivationTests ${SRC})

target_link_libraries(ActivationTests gtest Argentium)

install(TARGETS ActivationTests DESTINATION bin)