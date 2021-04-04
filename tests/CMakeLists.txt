file(GLOB SRC
	"*.hpp"
	"*.cpp"
)

add_subdirectory(googletest)
include_directories(../include)


add_library(GTest ${SRC})

target_link_libraries(GTest gtest)


add_subdirectory(dataset_tests)
add_subdirectory(activation_test)



