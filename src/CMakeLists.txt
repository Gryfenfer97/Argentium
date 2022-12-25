file(GLOB HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/Argentium/*.hpp")
file(GLOB_RECURSE SRC "./*.cpp")


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(argentium ${SRC} ${HEADER_LIST})

target_include_directories(argentium PUBLIC ../include)


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
  set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -Wpedantic -Werror -Wshadow")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -Wall -Wextra -Wpedantic")
endif()

set_target_properties(
  argentium
    PROPERTIES
    	LINKER_LANGUAGE CXX
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADERS}
)
