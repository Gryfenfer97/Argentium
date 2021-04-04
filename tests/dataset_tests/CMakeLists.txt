file(GLOB SRC
    "*.hpp"
    "*.cpp"
)

include_directories(.. tests)

add_executable(DatasetTests ${SRC})

target_link_libraries(DatasetTests gtest Argentium)

install(TARGETS DatasetTests DESTINATION bin)