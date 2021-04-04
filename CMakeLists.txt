cmake_minimum_required(VERSION 3.16)
project(Argentium)

file(GLOB_RECURSE SRC "${Argentium_SOURCE_DIR}/src/*.cpp" "${Argentium_SOURCE_DIR}/include/Argentium/*.hpp")

add_library(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PUBLIC "include")

add_subdirectory(tests)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message("-- Compile in ${CMAKE_BUILD_TYPE} mode")
if(MSVC)
    message("-- CMake run for msvc")
    string(REPLACE "/Ob2" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "/MD" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "/WX-" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_RELEASE})
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D \"NDEBUG\" /O2 /Ot /Ob3 /Oi /Oy /GR /GT /GL /GS- /MT /fp:fast /W4 /WX")
    string(REPLACE "/MDd" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    string(REPLACE "/W3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "/W3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_DEBUG})
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /sdl /GR /GS /Zi /MTd /W4 /WX")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
endif()



set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    	LINKER_LANGUAGE CXX
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)
